"""The latrec shim must expose libe3's registered stage ids, cost nothing when
the installed libe3py was built without the recorder, and write real records
when it was built with it."""
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from e3interface import latrec  # noqa: E402

_CATALOG = {
    "DECODE_E3SM_BEGIN": ("LATREC_DECODE_E3SM_BEGIN", 0x1A),
    "DECODE_E3SM_DONE": ("LATREC_DECODE_E3SM_DONE", 0x1B),
    "ENCODE_E3SM_BEGIN": ("LATREC_ENCODE_E3SM_BEGIN", 0x18),
    "ENCODE_E3SM_DONE": ("LATREC_ENCODE_E3SM_DONE", 0x19),
    "PROCESS_DONE": ("LATREC_PROCESS_DONE", 0x40),
    "CREATE_OUTPUT": ("LATREC_CREATE_OUTPUT", 0x41),
    "APPLY_POLICY_DONE": ("LATREC_APPLY_POLICY_DONE", 0x42),
}


@pytest.mark.parametrize("attr", sorted(_CATALOG))
def test_stage_ids_are_libe3_registered_ids(attr):
    libe3py = pytest.importorskip("libe3py")
    libe3_name, expected = _CATALOG[attr]
    assert getattr(libe3py, libe3_name) == expected
    assert getattr(latrec, attr) == expected


@pytest.mark.skipif(latrec.ENABLED, reason="libe3py was built with the recorder")
def test_recorder_off_is_a_no_op():
    assert latrec.stamp(1, 0x40, 2, 3) is None
    latrec.ctx_set(9)
    assert latrec.ctx() == 0
    assert latrec.seq_next() == 0
    assert latrec.open_ring("dapp.test") is None


@pytest.mark.skipif(not latrec.ENABLED, reason="libe3py was built without the recorder")
def test_ring_records_stamps_in_libe3_format(tmp_path):
    latrec.set_output_dir(str(tmp_path))
    try:
        latrec.open_ring("dapp.test")
        seq = latrec.seq_next()
        latrec.ctx_set(seq)
        assert latrec.ctx() == seq
        latrec.stamp(seq, latrec.DECODE_E3SM_BEGIN, 11, latrec.PDU_INDICATION)
        latrec.stamp(seq, latrec.PROCESS_DONE, 7)
    finally:
        latrec.set_output_dir("")

    rings = [p for p in tmp_path.iterdir() if p.name.startswith("dapp.test.")]
    assert len(rings) == 1
    data = rings[0].read_bytes()
    assert struct.unpack_from("<I", data, 0)[0] == 0x31524C41
    recs = []
    for i in range(2):
        sc, t_ns, aux, aux2 = struct.unpack_from("<QQQQ", data, 4096 + 32 * i)
        recs.append((sc >> 56, sc & 0xFFFFFFFFFFFF, aux, aux2, t_ns))
    assert [r[0] for r in recs] == [0x1A, 0x40]
    assert all(r[1] == seq and r[4] > 0 for r in recs)
    assert recs[0][2:4] == (11, latrec.PDU_INDICATION)
    assert recs[1][2] == 7
