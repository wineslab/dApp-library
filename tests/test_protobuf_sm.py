"""Protobuf E3SM codec round-trips (spectrum + simple).

Exercises the dApp-side protobuf encode/decode in isolation (no E3AP / libe3):
the wire bytes are standard proto3, so these also pin the interop contract with
the gNB's protobuf-c encoder that shares the same .proto field numbers.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

pytest.importorskip("google.protobuf")

from spectrum.spectrum_dapp import SpectrumSharingDApp  # noqa: E402
from simple.simple_dapp import SimpleDApp  # noqa: E402


def _spectrum_codec():
    d = object.__new__(SpectrumSharingDApp)
    d.encoding_method = "protobuf"
    d._init_spectrum_encoder()
    return d


def _simple_codec():
    d = object.__new__(SimpleDApp)
    d.encoding_method = "protobuf"
    d._init_simple_encoder()
    return d


def test_spectrum_prb_blacklist_control_roundtrip():
    d = _spectrum_codec()
    inner = {"blacklistedPRBs": [0, 5, 272], "samplingThreshold": 7, "validityPeriod": 60}
    encoded = d._encode_dapp_control_envelope(
        type_value="prbBlacklist", payload_key="prbBlacklistControl",
        inner_type="Spectrum-PRBBlacklistControl", inner=inner,
    )
    assert isinstance(encoded, bytes) and encoded  # non-empty proto3 bytes

    dec = d._decode_envelope(
        encoded, msg_type="Spectrum-DAppControlData",
        type_field="controlType", payload_field="controlPayload",
        inner_type_map={"prbBlacklistControl": "Spectrum-PRBBlacklistControl"},
        type_by_key={"prbBlacklistControl": "prbBlacklist"},
    )
    assert dec["payload_key"] == "prbBlacklistControl"
    assert dec["type"] == "prbBlacklist"
    assert dec["payload"]["blacklistedPRBs"] == [0, 5, 272]
    assert dec["payload"]["samplingThreshold"] == 7
    assert dec["payload"]["validityPeriod"] == 60


def test_spectrum_empty_prb_blacklist_roundtrip():
    """A clear (empty PRB list) must round-trip as an empty list, not vanish."""
    d = _spectrum_codec()
    encoded = d._encode_dapp_control_envelope(
        type_value="prbBlacklist", payload_key="prbBlacklistControl",
        inner_type="Spectrum-PRBBlacklistControl", inner={"blacklistedPRBs": []},
    )
    dec = d._decode_envelope(
        encoded, msg_type="Spectrum-DAppControlData",
        type_field="controlType", payload_field="controlPayload",
        inner_type_map={"prbBlacklistControl": "Spectrum-PRBBlacklistControl"},
        type_by_key={"prbBlacklistControl": "prbBlacklist"},
    )
    assert dec["payload_key"] == "prbBlacklistControl"
    assert dec["payload"].get("blacklistedPRBs", []) == []


def test_spectrum_xapp_config_control_decode():
    d = _spectrum_codec()
    # Build an xApp ConfigControl exactly as the gNB would emit it.
    x = d._spectrum_pb_new("Spectrum-XAppControlData")
    from google.protobuf import json_format
    json_format.ParseDict({"configControl": {"noiseFloorThreshold": -50, "enable": True}}, x)
    wire = x.SerializeToString()

    dec = d._decode_xapp_control_envelope(wire)
    assert dec["payload_key"] == "configControl"
    assert dec["payload"]["noiseFloorThreshold"] == -50
    assert dec["payload"]["enable"] is True


def test_spectrum_sensing_indication_decode():
    d = _spectrum_codec()
    si = d._spectrum_pb_new("Spectrum-SensingIndication")
    si.timestamp = 123456789
    si.sfn = 10
    si.slot = 8
    si.shm_name = b"/e3_l2_sensing"
    si.shm_write_idx = 3
    si.n_ranges = 0  # a slot with zero ranges must still decode (not be dropped)
    wire = si.SerializeToString()

    msg = d._decode_spectrum_message("Spectrum-SensingIndication", wire)
    assert msg["sfn"] == 10 and msg["slot"] == 8
    assert msg["shmWriteIdx"] == 3
    assert msg["nRanges"] == 0            # proto3 default present, not omitted
    assert msg["shmName"] == b"/e3_l2_sensing"  # bytes kept as bytes


def test_simple_indication_and_control_roundtrip():
    d = _simple_codec()
    ind = d._encode_simple_message("Simple-Indication", {"data1": 0, "timestamp": 42})
    got = d._decode_simple_message("Simple-Indication", ind)
    assert got["data1"] == 0          # data1 == 0 must survive (proto3 default)
    assert got["timestamp"] == 42

    ctrl = d._encode_simple_message("Simple-Control", {"samplingThreshold": 9})
    got_ctrl = d._decode_simple_message("Simple-Control", ctrl)
    assert got_ctrl["samplingThreshold"] == 9
