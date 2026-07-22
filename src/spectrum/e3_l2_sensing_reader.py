"""Reader for the OAI gNB ``/e3_l2_sensing`` POSIX shm ring.

The Spectrum SM (RF=1) publishes a ``Spectrum-SensingIndication`` per MAC
sensing snapshot; the sensing ranges themselves are written out-of-band into
this ring and referenced by ``(shmWriteIdx, nRanges)``. Layout mirrors
``openair2/E3AP/service_models/spectrum_sm/spectrum_sensing_ring.{h,c}``:

    header: version, slot_count, slot_stride, max_ranges, range_size (u32 x5, 64B)
    slot:   sfn, slot, beam, n_ranges (u16 x4) | timestamp_ns (u64) |
            seq, _pad (u32 x2) | sensing_range_t ranges[max_ranges]
    range:  start_symbol, num_symbols, rb_start, rb_size (int32 x4)

The writer stamps ``(sfn, slot)`` last, so the reader validates the slot tag
against the indication as a torn/wrap guard.
"""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass
from typing import Optional

SHM_NAME = "/e3_l2_sensing"
SHM_PATH = f"/dev/shm{SHM_NAME}"

_HEADER_FMT = "<5I"                # version, slot_count, slot_stride, max_ranges, range_size
_HEADER_SIZE = 64                  # 16 x u32 (11 reserved), matches the writer
_SLOT_HDR_FMT = "<4HQII"           # sfn, slot, beam, n_ranges, timestamp_ns, seq, _pad
_SLOT_HDR_SIZE = struct.calcsize(_SLOT_HDR_FMT)  # 24
_RANGE_FMT = "<4i"                 # start_symbol, num_symbols, rb_start, rb_size
_RANGE_SIZE = struct.calcsize(_RANGE_FMT)        # 16


@dataclass(frozen=True)
class SensingRange:
    """One sensing window — a (symbol-span, PRB-span) rectangle the MAC
    scheduler committed as free for sensing."""
    start_symbol: int
    num_symbols: int
    rb_start: int
    rb_size: int


class E3L2SensingReader:
    """Mmap-backed reader of ``/e3_l2_sensing``. Single-thread per instance."""

    def __init__(self, shm_path: str = SHM_PATH):
        self._shm_path = shm_path
        self._mmap: Optional[mmap.mmap] = None
        self._ino: Optional[int] = None
        self._slot_count = 0
        self._slot_stride = 0
        self._max_ranges = 0

    def open(self) -> None:
        if self._mmap is not None:
            return
        with open(self._shm_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(0)
            self._mmap = mmap.mmap(f.fileno(), size, prot=mmap.PROT_READ)
            self._ino = os.fstat(f.fileno()).st_ino
        version, slot_count, slot_stride, max_ranges, range_size = struct.unpack_from(
            _HEADER_FMT, self._mmap, 0
        )
        expected_stride = _SLOT_HDR_SIZE + max_ranges * _RANGE_SIZE
        mmap_size = self._mmap.size()
        if (version != 1 or slot_count == 0 or range_size != _RANGE_SIZE
                or slot_stride != expected_stride
                or mmap_size < _HEADER_SIZE + slot_count * slot_stride):
            self.close()
            raise RuntimeError(
                f"/e3_l2_sensing header inconsistent: version={version} "
                f"slot_count={slot_count} slot_stride={slot_stride} "
                f"max_ranges={max_ranges} range_size={range_size} "
                f"mmap_size={mmap_size}"
            )
        self._slot_count = slot_count
        self._slot_stride = slot_stride
        self._max_ranges = max_ranges

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        self._ino = None

    @property
    def is_open(self) -> bool:
        return self._mmap is not None

    def reopen_if_stale(self) -> bool:
        if self._mmap is None:
            return False
        try:
            live_ino = os.stat(self._shm_path).st_ino
        except OSError:
            live_ino = None
        if live_ino == self._ino:
            return False
        self.close()
        try:
            self.open()
        except (OSError, ValueError, RuntimeError, struct.error):
            # gNB gone or mid-recreate (0-byte / short / inconsistent segment);
            # caller's lazy-open retries later.
            pass
        return True

    def read_ranges(self, write_idx: int, sfn: int, slot: int, n_ranges: int) -> list:
        """Return the sensing ranges at ``write_idx`` if the slot tag matches
        ``(sfn, slot)``; ``[]`` on mismatch (torn/wrap) or out-of-range.

        The writer stamps ``(sfn, slot)`` last and bumps ``seq``, so we snapshot
        the tag + seq, copy the ranges, then re-read and require both unchanged —
        otherwise the producer wrapped onto this slot mid-copy and the ranges are
        validated against a stale tag.
        """
        self.reopen_if_stale()
        if self._mmap is None or not (0 <= write_idx < self._slot_count):
            return []
        base = _HEADER_SIZE + write_idx * self._slot_stride
        try:
            r_sfn, r_slot, _beam, r_n, _ts, seq0, _pad = struct.unpack_from(
                _SLOT_HDR_FMT, self._mmap, base
            )
            if r_sfn != (sfn & 0xFFFF) or r_slot != (slot & 0xFFFF):
                return []
            count = min(int(n_ranges), int(r_n), self._max_ranges)
            out = []
            off = base + _SLOT_HDR_SIZE
            for _ in range(count):
                ss, ns, rbs, rbsz = struct.unpack_from(_RANGE_FMT, self._mmap, off)
                out.append(SensingRange(ss, ns, rbs, rbsz))
                off += _RANGE_SIZE
            # Re-read the tag+seq: if the producer overwrote this slot during
            # the copy, discard the possibly-torn result.
            r_sfn2, r_slot2, _b2, _n2, _ts2, seq1, _p2 = struct.unpack_from(
                _SLOT_HDR_FMT, self._mmap, base
            )
        except struct.error:
            return []
        if r_sfn2 != r_sfn or r_slot2 != r_slot or seq1 != seq0:
            return []
        return out

    def __enter__(self) -> "E3L2SensingReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
