"""Spectrum SM producer timestamps on reports and xApp controls, and the gNB's
validSymbolMask in the detector window."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

pytest.importorskip("google.protobuf")

from spectrum.spectrum_dapp import SpectrumSharingDApp  # noqa: E402
from spectrum.e3_l2_sensing_reader import SensingRange  # noqa: E402


def _codec(encoding):
    d = object.__new__(SpectrumSharingDApp)
    d.encoding_method = encoding
    d._init_spectrum_encoder()
    return d


def _decode_report(d, wire):
    return d._decode_envelope(
        wire, msg_type="Spectrum-DAppReportData",
        type_field="reportType", payload_field="reportPayload",
        inner_type_map={"prbBlacklistReport": "Spectrum-PRBBlacklistReport"},
        type_by_key={"prbBlacklistReport": "prbBlacklist"},
    )


@pytest.mark.parametrize("encoding", ["asn1", "json", "protobuf"])
def test_report_carries_a_realtime_producer_timestamp(encoding):
    d = _codec(encoding)
    before = time.time_ns()
    dec = _decode_report(d, d.create_prb_blacklist_report([3, 4]))
    after = time.time_ns()
    assert dec["payload_key"] == "prbBlacklistReport"
    assert dec["payload"]["blacklistedPRBs"] == [3, 4]
    assert before <= dec["timestamp"] <= after


def test_stamped_xapp_control_keeps_its_payload_key_on_protobuf():
    d = _codec("protobuf")
    msg = d._spectrum_pb_new("Spectrum-XAppControlData")
    from google.protobuf import json_format
    json_format.ParseDict({"timestamp": 123, "prbBlockedControl": {"blockedPRBs": [7]}}, msg)
    env = d._decode_xapp_control_envelope(msg.SerializeToString())
    assert env["payload_key"] == "prbBlockedControl"
    assert env["payload"]["blockedPRBs"] == [7]
    assert env["timestamp"] == 123


def test_unstamped_xapp_control_decodes_with_no_timestamp():
    d = _codec("protobuf")
    msg = d._spectrum_pb_new("Spectrum-XAppControlData")
    from google.protobuf import json_format
    json_format.ParseDict({"prbBlockedControl": {"blockedPRBs": [7]}}, msg)
    env = d._decode_xapp_control_envelope(msg.SerializeToString())
    assert env["timestamp"] is None


def _detector_input(valid_symbol_mask):
    d = object.__new__(SpectrumSharingDApp)
    d.num_consecutive_subcarriers_for_prb = 12
    mag = np.ones((14, 24), dtype=np.float32)
    ranges = (SensingRange(start_symbol=0, num_symbols=14, rb_start=0, rb_size=2),)
    return d._build_detector_input(mag, ranges, valid_symbol_mask)


def test_invalid_symbols_are_dropped_from_the_detector_window():
    out, col_keep = _detector_input(0b11111100000000)
    assert col_keep.all() and (out == 1.0).all()
    assert _detector_input(0) is None


def test_default_mask_keeps_every_symbol():
    assert _detector_input(0x3FFF) is not None
