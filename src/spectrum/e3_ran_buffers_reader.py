"""Reader for the OAI gNB ``/e3_ran_buffers`` POSIX shm region.

The gNB publishes UL slot IQ samples in cuBB-compatible layout: a fixed
header followed by two double-buffered FH (frequency-domain post-FFT)
regions of N rows each. Each row is

    ``[ant=4][sym=14][prb=273][sc=12][I,Q=2]`` in FP16

i.e. 4 antennas × 14 OFDM symbols × 273 PRBs × 12 subcarriers × 2 half-
precision floats. The OAI-L1-KPM SM (RAN function 2) emits a per-slot
indication whose ``protocol_data`` references this shm. The on-wire
shape depends on which libe3 channel the dApp registered on:

  - JSON channel: inline JSON object
        ``{"iq_samples":{"shm_name":"/e3_ran_buffers",
                         "fh_buffer_index":B,"fh_write_index":W},
          "timestamp":...,"sfn":...,"slot":...,"cell_id":...,"n_rx_ant":...}``

  - ASN.1 channel: APER-encoded ``L1KPM-Indication`` (schema in
        ``defs/e3sm_oai_l1_kpm.asn``)

Both decode to the same :class:`SlotPointer`; ``SlotPointer.from_bytes``
sniffs the first byte (``{`` vs. anything else) and dispatches.

The reader mmaps the shm once at startup (read-only) and on each
indication peels the (B, W) row, unpacks FP16 → float32, and rescales
by ``1/fp16_beta`` so the values come out in the same Q1.15-equivalent
magnitude range the legacy int16 path used. That keeps existing dApp
threshold calibration valid without a separate retune.

Layout constants mirror ``openair2/E3AP/service_models/oai_l1_kpm_sm/
e3_ran_buffers.c`` (``E3_RB_N_*``) and ``aerial-cuda-accelerated-ran/
cuPHY-CP/data_lake/e3_agent.hpp``.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
from dataclasses import dataclass
from typing import Optional

import asn1tools
import numpy as np


SHM_NAME = "/e3_ran_buffers"
SHM_PATH = f"/dev/shm{SHM_NAME}"

# Layout constants (must match OAI's e3_ran_buffers.c)
N_ANTS = 4
N_SYMBOLS = 14
N_PRBS = 273
N_SC_PER_PRB = 12
N_SC_PER_SLOT = N_PRBS * N_SC_PER_PRB  # 3276
BYTES_PER_SAMPLE = 4  # 2 × FP16

# Row = one slot's worth of FP16 IQ
ROW_BYTES = N_ANTS * N_SYMBOLS * N_SC_PER_SLOT * BYTES_PER_SAMPLE  # 733824

# Header: e3_ran_buffers_header_t — 9 named uint32_t + 7 reserved = 16 × uint32_t
_HEADER_FMT = "<9I7I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
assert _HEADER_SIZE == 64, f"Header expected 64 bytes, got {_HEADER_SIZE}"


# ----------------------------------------------------------------------------
# OAI-L1-KPM ASN.1 schema (mirror of OAI's e3sm_oai_l1_kpm.asn).
#
# The dApp gets one of two inner-payload formats per indication depending
# on which libe3 channel it registered on:
#   - JSON channel: payload is a JSON object (legacy / aerial format)
#   - ASN.1 channel: payload is APER-encoded L1KPM-Indication
#
# asn1tools' compile_files is heavy (~50 ms on cold disk), so the compiler
# is built once on first use and cached at module scope.
# ----------------------------------------------------------------------------

_OAI_L1_KPM_SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), "defs", "e3sm_oai_l1_kpm.asn"
)
_oai_l1_kpm_compiler = None


def _get_oai_l1_kpm_compiler():
    """Return a cached asn1tools compiler for L1KPM-Indication."""
    global _oai_l1_kpm_compiler
    if _oai_l1_kpm_compiler is None:
        _oai_l1_kpm_compiler = asn1tools.compile_files(
            _OAI_L1_KPM_SCHEMA_PATH, codec="per"
        )
    return _oai_l1_kpm_compiler


@dataclass(frozen=True)
class SlotPointer:
    """Decoded shm coordinates from an L1-KPM (RF=2) indication.

    ``valid_symbol_mask`` is the gNB's 14-bit UL-symbol bitmap (bit s set =
    symbol s carries genuine off-air UL); ``0x3FFF`` when the field is absent.
    Sensing ranges do not ride here — they arrive on the RF=1 Spectrum SM (see
    :mod:`spectrum.e3_l2_sensing_reader`).
    """

    fh_buffer_index: int
    fh_write_index: int
    sfn: int
    slot: int
    cell_id: int
    n_rx_ant: int
    timestamp_ns: int
    valid_symbol_mask: int = 0x3FFF

    @classmethod
    def from_json_bytes(cls, data: bytes) -> Optional["SlotPointer"]:
        """Parse an L1-KPM indication payload as JSON.

        Returns None if the payload is not the expected shape (e.g. a
        future SM emits a different telemetry id with a different schema).
        """
        try:
            obj = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

        iq = obj.get("iq_samples")
        if not isinstance(iq, dict):
            return None

        try:
            return cls(
                fh_buffer_index=int(iq["fh_buffer_index"]),
                fh_write_index=int(iq["fh_write_index"]),
                sfn=int(obj.get("sfn", 0)),
                slot=int(obj.get("slot", 0)),
                cell_id=int(obj.get("cell_id", 0)),
                n_rx_ant=int(obj.get("n_rx_ant", N_ANTS)),
                timestamp_ns=int(obj.get("timestamp", 0)),
                valid_symbol_mask=int(obj.get("valid_symbol_mask", 0x3FFF)),
            )
        except (KeyError, TypeError, ValueError):
            return None

    @classmethod
    def from_asn1_bytes(cls, data: bytes) -> Optional["SlotPointer"]:
        """Parse an L1-KPM indication payload as APER bytes.

        Schema: L1KPM-Indication from defs/e3sm_oai_l1_kpm.asn — kept in
        lockstep with the OAI APER encoder. Returns None if the buffer
        doesn't decode against the schema or the required iqSamplesRef
        sub-SEQUENCE is missing.
        """
        try:
            decoded = _get_oai_l1_kpm_compiler().decode("L1KPM-Indication", data)
        except Exception:
            return None

        iq = decoded.get("iqSamplesRef")
        if not isinstance(iq, dict):
            return None

        try:
            return cls(
                fh_buffer_index=int(iq["fhBufferIndex"]),
                fh_write_index=int(iq["fhWriteIndex"]),
                sfn=int(decoded["sfn"]),
                slot=int(decoded["slot"]),
                cell_id=int(decoded.get("cellId", 0)),
                n_rx_ant=int(decoded.get("nRxAnt", N_ANTS)),
                timestamp_ns=int(decoded.get("timestamp", 0)),
                # validSymbolMask is INTEGER(0..16383) OPTIONAL; 0 is legal
                # (all symbols DL/guard). `or 0x3FFF` would flip it to all-valid.
                valid_symbol_mask=int(decoded.get("validSymbolMask", 0x3FFF)),
            )
        except (KeyError, TypeError, ValueError):
            return None

    @classmethod
    def from_bytes(cls, data: bytes) -> Optional["SlotPointer"]:
        """Auto-detect JSON vs APER and dispatch to the matching parser.

        Discriminator is the first byte: JSON payloads start with ``{``
        (0x7B), APER payloads start with the SEQUENCE's OPTIONAL preamble
        (four OPTIONALs in L1KPM-Indication). 0x7B is not a valid APER
        preamble byte here, so a one-byte sniff is sufficient.
        """
        if not data:
            return None
        if data[0] == 0x7B:  # '{'
            return cls.from_json_bytes(data)
        return cls.from_asn1_bytes(data)


@dataclass(frozen=True)
class _Header:
    fh_buffer_size: int
    num_fh_rows: int


class E3RanBuffersReader:
    """Mmap-backed reader of ``/e3_ran_buffers``.

    Thread-safe for reads from a single thread per instance. Multiple
    threads should use multiple readers (mmap is cheap; the kernel
    page cache shares the underlying pages).
    """

    def __init__(self, fp16_beta: float = 1.0 / 2048.0, shm_path: str = SHM_PATH):
        self._shm_path = shm_path
        self._fp16_beta = float(fp16_beta)
        if self._fp16_beta <= 0.0:
            raise ValueError(f"fp16_beta must be positive, got {fp16_beta}")
        # 1/beta — multiply FP16 floats by this to recover Q1.15-equivalent magnitudes.
        self._rescale = 1.0 / self._fp16_beta
        self._mmap: Optional[mmap.mmap] = None
        self._header: Optional[_Header] = None
        self._fh_base_offset: int = _HEADER_SIZE
        self._ino: Optional[int] = None

    def open(self) -> None:
        """Map the shm region. Raises FileNotFoundError if OAI hasn't
        published yet (the gNB creates the region on first UL slot)."""
        if self._mmap is not None:
            return

        with open(self._shm_path, "rb") as f:
            # mmap requires size — use the file's reported length. POSIX
            # shm regions are visible as files under /dev/shm so this works.
            f.seek(0, 2)
            size = f.tell()
            f.seek(0)
            self._mmap = mmap.mmap(f.fileno(), size, prot=mmap.PROT_READ)
            self._ino = os.fstat(f.fileno()).st_ino

        # Parse header — first 9 uint32_t are: version, fh_buffer_size,
        # pusch_buffer_size, hest_buffer_size, num_fh_samples, num_fh_rows,
        # num_pusch_rows, num_hest_rows, max_hest_samples_per_row.
        version, fh_buffer_size, _, _, _, num_fh_rows, _, _, _, *_ = struct.unpack_from(
            _HEADER_FMT, self._mmap, 0
        )
        if version != 1:
            self.close()
            raise RuntimeError(f"unsupported /e3_ran_buffers header version {version}")
        if num_fh_rows == 0 or fh_buffer_size != num_fh_rows * ROW_BYTES:
            self.close()
            raise RuntimeError(
                f"/e3_ran_buffers header inconsistent: fh_buffer_size={fh_buffer_size}"
                f" num_fh_rows={num_fh_rows} expected_row_bytes={ROW_BYTES}"
            )
        self._header = _Header(fh_buffer_size=fh_buffer_size, num_fh_rows=num_fh_rows)

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        self._header = None
        self._ino = None

    def reopen_if_stale(self) -> bool:
        """Remap if the gNB recreated the shm region (restart → new inode).

        The gNB ``shm_unlink``s and recreates ``/e3_ran_buffers`` on init, so a
        stale mapping keeps returning frozen IQ with in-range indices and no
        error. Returns True if it remapped.
        """
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
            # gNB gone or mid-recreate: the shm_open→ftruncate window exposes a
            # 0-byte / short / version-inconsistent segment. Caller's lazy-open
            # retries later; don't let it crash the read loop.
            pass
        return True

    @property
    def is_open(self) -> bool:
        return self._mmap is not None

    @property
    def num_fh_rows(self) -> int:
        return self._header.num_fh_rows if self._header else 0

    def read_slot(self, p: SlotPointer) -> Optional[np.ndarray]:
        """Read the slot at ``(p.fh_buffer_index, p.fh_write_index)``.

        Returns a float32 ndarray of shape ``[ant][sym][sc_per_slot][2]``
        where the trailing dim is ``[I, Q]``. Values are rescaled by
        ``1/fp16_beta`` so magnitudes match the legacy int16 path.
        ``ant`` is clamped to ``p.n_rx_ant`` (typically 1–4).

        Returns None if the reader is closed or the indices are out of
        range — caller should treat as a read-drop.
        """
        self.reopen_if_stale()
        if self._mmap is None or self._header is None:
            return None
        if p.fh_buffer_index not in (0, 1):
            return None
        if not (0 <= p.fh_write_index < self._header.num_fh_rows):
            return None

        offset = (
            self._fh_base_offset
            + p.fh_buffer_index * self._header.fh_buffer_size
            + p.fh_write_index * ROW_BYTES
        )

        # Build a zero-copy uint16 view over the FP16 bit pattern, then
        # reinterpret as float16 (numpy supports IEEE 754 half natively).
        # The view is read-only because the mmap is PROT_READ.
        raw = np.frombuffer(self._mmap, dtype=np.uint16, count=ROW_BYTES // 2, offset=offset)
        fp16 = raw.view(np.float16).reshape(N_ANTS, N_SYMBOLS, N_SC_PER_SLOT, 2)

        # Upcast to float32 and rescale. Trim to n_rx_ant — antennas beyond
        # what the gNB has are zero-filled by the writer; keeping them as
        # zeros would dilute energy if a caller summed across antennas.
        ants = max(1, min(int(p.n_rx_ant), N_ANTS))
        return (fp16[:ants].astype(np.float32) * self._rescale).reshape(
            ants, N_SYMBOLS, N_SC_PER_SLOT, 2
        )

    def __enter__(self) -> "E3RanBuffersReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
