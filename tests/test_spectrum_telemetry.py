#!/usr/bin/env python3
"""Tests for the RF=2/RF=1 telemetry rework (aligned to the gNB)."""
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asn1tools

from spectrum.adaptive_noise_floor import AdaptiveNoiseFloor
from spectrum.e3_l2_sensing_reader import (
    E3L2SensingReader, _HEADER_SIZE, _SLOT_HDR_FMT, _RANGE_FMT,
)
from spectrum.e3_ran_buffers_reader import SlotPointer

_L1_ASN = os.path.join(os.path.dirname(__file__), "..", "src", "spectrum", "defs", "e3sm_oai_l1_kpm.asn")


def test_l1_kpm_asn1_roundtrip_valid_symbol_mask():
    c = asn1tools.compile_files(_L1_ASN, codec="per")
    msg = {
        "iqSamplesRef": {"shmName": b"/e3_ran_buffers", "fhBufferIndex": 1, "fhWriteIndex": 9},
        "timestamp": 42, "sfn": 10, "slot": 3, "cellId": 0, "nRxAnt": 4,
        "validSymbolMask": 0x2AAA,
    }
    p = SlotPointer.from_bytes(c.encode("L1KPM-Indication", msg))
    assert p is not None
    assert p.fh_buffer_index == 1 and p.fh_write_index == 9
    assert p.valid_symbol_mask == 0x2AAA
    assert not hasattr(p, "sensing_ranges")


def test_slotpointer_json_valid_symbol_mask():
    payload = json.dumps({
        "iq_samples": {"shm_name": "/e3_ran_buffers", "fh_buffer_index": 0, "fh_write_index": 3},
        "sfn": 1, "slot": 2, "cell_id": 0, "n_rx_ant": 4, "valid_symbol_mask": 0x1555,
    }).encode()
    p = SlotPointer.from_bytes(payload)
    assert p is not None and p.fh_write_index == 3 and p.valid_symbol_mask == 0x1555


def test_l2_sensing_ring_parse(tmp_path):
    slot_count, max_ranges, range_size = 2, 4, 16
    slot_stride = struct.calcsize(_SLOT_HDR_FMT) + max_ranges * range_size
    buf = bytearray(_HEADER_SIZE + slot_count * slot_stride)
    struct.pack_into("<5I", buf, 0, 1, slot_count, slot_stride, max_ranges, range_size)
    base = _HEADER_SIZE  # slot 0
    struct.pack_into(_SLOT_HDR_FMT, buf, base, 7, 5, 0, 2, 123, 0, 0)  # sfn=7 slot=5 n_ranges=2
    struct.pack_into(_RANGE_FMT, buf, base + struct.calcsize(_SLOT_HDR_FMT), 2, 4, 10, 6)
    struct.pack_into(_RANGE_FMT, buf, base + struct.calcsize(_SLOT_HDR_FMT) + range_size, 0, 14, 50, 3)

    path = tmp_path / "e3_l2_sensing"
    path.write_bytes(buf)
    r = E3L2SensingReader(shm_path=str(path))
    r.open()
    ranges = r.read_ranges(write_idx=0, sfn=7, slot=5, n_ranges=2)
    assert len(ranges) == 2
    assert (ranges[0].start_symbol, ranges[0].num_symbols, ranges[0].rb_start, ranges[0].rb_size) == (2, 4, 10, 6)
    # (sfn, slot) mismatch → torn/wrap guard returns nothing
    assert r.read_ranges(write_idx=0, sfn=7, slot=6, n_ranges=2) == []
    r.close()


def test_adaptive_noise_floor_masked():
    nf = AdaptiveNoiseFloor(n=4, x=3)
    mask = np.array([True, False, True, True])
    for _ in range(3):
        nf.update(np.array([10.0, 999.0, 10.0, 10.0], dtype=np.float32), mask)
    floor = nf.get_noise_floor()
    assert floor[0] == 10.0 and floor[2] == 10.0 and floor[3] == 10.0
    assert np.isinf(floor[1])  # always-masked bin never enters the median


def test_adaptive_noise_floor_unmasked_median():
    nf = AdaptiveNoiseFloor(n=2, x=2)
    nf.update(np.array([1.0, 2.0], dtype=np.float32))
    nf.update(np.array([3.0, 4.0], dtype=np.float32))
    assert list(nf.get_noise_floor()) == [2.0, 3.0]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
