"""Tests for the sensing-policy control: schema mask helpers + ASN.1/JSON
round-trips of Spectrum-SensingPolicyControl through the Spectrum-DAppControlData
envelope.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectrum.spectrum_dapp import (  # noqa: E402
    make_uniform_mask,
    make_symbol_range_mask,
    make_periodic_toggle_callback,
)


# ----------------------------------------------------------------------
# Pure-Python helpers
# ----------------------------------------------------------------------

class TestMakeUniformMask:
    def test_basic(self):
        assert make_uniform_mask(4, 0x3F80) == [0x3F80] * 4

    def test_rejects_out_of_range_bitmap(self):
        # Previously silently truncated to 14 bits; now raises so the
        # caller learns about the mistake instead of getting a mask
        # they didn't ask for.
        with pytest.raises(ValueError):
            make_uniform_mask(2, 0xFFFFFFFF)
        with pytest.raises(ValueError):
            make_uniform_mask(2, 0x4000)
        with pytest.raises(ValueError):
            make_uniform_mask(2, -1)

    def test_zero_mask(self):
        assert make_uniform_mask(5, 0) == [0, 0, 0, 0, 0]

    def test_max_valid_mask(self):
        assert make_uniform_mask(3, 0x3FFF) == [0x3FFF, 0x3FFF, 0x3FFF]

    def test_rejects_non_positive_slots(self):
        with pytest.raises(ValueError):
            make_uniform_mask(0, 0)
        with pytest.raises(ValueError):
            make_uniform_mask(-1, 0)


class TestMakeSymbolRangeMask:
    def test_uniform_range(self):
        # symbols 7..13 → bits 7..13 = 0x3F80
        out = make_symbol_range_mask(n_slots=4, start_sym=7, num_sym=7)
        assert out == [0x3F80] * 4

    def test_target_slots_subset(self):
        out = make_symbol_range_mask(n_slots=4, start_sym=0, num_sym=4,
                                     target_slots=[1, 3])
        assert out == [0, 0x000F, 0, 0x000F]

    def test_range_out_of_bounds(self):
        with pytest.raises(ValueError):
            make_symbol_range_mask(n_slots=4, start_sym=10, num_sym=5)
        with pytest.raises(ValueError):
            make_symbol_range_mask(n_slots=4, start_sym=-1, num_sym=4)
        with pytest.raises(ValueError):
            make_symbol_range_mask(n_slots=4, start_sym=0, num_sym=0)

    def test_target_slot_out_of_range(self):
        with pytest.raises(ValueError):
            make_symbol_range_mask(n_slots=4, start_sym=0, num_sym=4,
                                   target_slots=[10])


class TestPeriodicToggleCallback:
    def test_first_call_activates(self):
        """First call activates regardless of ``now`` (None-sentinel
        last_toggle).  Operator intuition: 'active at startup'."""
        cb = make_periodic_toggle_callback(period_s=10.0, n_slots=20,
                                           mask_when_on=0x3F80)
        res = cb(now=0.0)
        assert res is not None
        send_now, mask, deactivate = res
        assert send_now is True
        assert mask == [0x3F80] * 20
        assert deactivate is False
        # Subsequent call within period_s of the first → no-op
        assert cb(now=5.0) is None

    def test_first_call_activates_with_realistic_monotonic(self):
        """Regression guard for the docstring claim 'active at startup'
        against a realistic time.monotonic() value (large positive).
        Pre-None-sentinel implementation passed test_first_call_activates
        only by cheating with ``now=0.0``; in production
        ``now=time.monotonic()`` is a large positive and the first call
        WAS already activating, but the documented semantic was
        inconsistent with the test.  This test locks the contract."""
        cb = make_periodic_toggle_callback(period_s=10.0, n_slots=20,
                                           mask_when_on=0x3F80)
        # First call with a realistic monotonic value: must activate.
        res = cb(now=12345.678)
        assert res == (True, [0x3F80] * 20, False)
        # Within 10s of first toggle → None.
        assert cb(now=12350.0) is None
        # > 10s elapsed → flip to deactivate.
        assert cb(now=12356.0) == (True, [0] * 20, True)

    def test_alternates_off_on(self):
        cb = make_periodic_toggle_callback(period_s=1.0, n_slots=3,
                                           mask_when_on=0x0001)
        # tick 1 (FIRST call, None sentinel): activate immediately
        assert cb(now=2.0) == (True, [0x0001, 0x0001, 0x0001], False)
        # tick 2: still within 1s of previous toggle → None
        assert cb(now=2.5) is None
        # tick 3: > 1s passed → deactivate
        assert cb(now=3.1) == (True, [0, 0, 0], True)
        # tick 4: still within 1s of previous toggle → None
        assert cb(now=3.5) is None
        # tick 5: > 1s passed → re-activate
        assert cb(now=4.2) == (True, [0x0001, 0x0001, 0x0001], False)

    def test_rejects_bad_args(self):
        with pytest.raises(ValueError):
            make_periodic_toggle_callback(period_s=0, n_slots=20, mask_when_on=0)
        with pytest.raises(ValueError):
            make_periodic_toggle_callback(period_s=1.0, n_slots=0, mask_when_on=0)
        with pytest.raises(ValueError):
            make_periodic_toggle_callback(period_s=1.0, n_slots=20,
                                          mask_when_on=0xFFFFFFFF)


# ----------------------------------------------------------------------
# ASN.1 round-trip
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def spectrum_asn():
    """asn1tools compiler for the spectrum schema."""
    asn1tools = pytest.importorskip("asn1tools")
    schema = os.path.join(
        os.path.dirname(__file__), "..", "src", "spectrum", "defs",
        "e3sm_spectrum.asn",
    )
    return asn1tools.compile_files(schema, codec="per")


def test_asn1_sensing_policy_roundtrip_uniform(spectrum_asn):
    mask = [0x3F80] * 20
    # The ASN.1 envelope is payload-only (no controlType field) to match the
    # merged gNB decoder; the CHOICE alternative is the sole discriminator.
    encoded = spectrum_asn.encode("Spectrum-DAppControlData", {
        "controlPayload": ("sensingPolicyControl", {
            "maskPerSlot": mask,
            "deactivate": False,
        }),
    })
    decoded = spectrum_asn.decode("Spectrum-DAppControlData", encoded)
    key, payload = decoded["controlPayload"]
    assert key == "sensingPolicyControl"
    assert payload["maskPerSlot"] == mask
    # asn1tools fills in BOOLEAN DEFAULT FALSE as False
    assert payload.get("deactivate", False) is False


def test_asn1_sensing_policy_roundtrip_deactivate(spectrum_asn):
    encoded = spectrum_asn.encode("Spectrum-DAppControlData", {
        "controlPayload": ("sensingPolicyControl", {
            "maskPerSlot": [0] * 20,
            "deactivate": True,
            "validityPeriod": 60,
        }),
    })
    decoded = spectrum_asn.decode("Spectrum-DAppControlData", encoded)
    key, payload = decoded["controlPayload"]
    assert key == "sensingPolicyControl"
    assert payload["maskPerSlot"] == [0] * 20
    assert payload["deactivate"] is True
    assert payload["validityPeriod"] == 60


def test_asn1_prb_blacklist_roundtrip(spectrum_asn):
    """The Spectrum-DAppControlPayload CHOICE uses a '...' marker so
    additional alternatives can be added later without changing the APER
    wire bit-width of existing variants.  This test pins the
    prbBlacklist alternative's encode/decode against the gNB wire naming."""
    encoded = spectrum_asn.encode("Spectrum-DAppControlData", {
        "controlPayload": ("prbBlacklistControl", {
            "blacklistedPRBs": [3, 7, 11],
        }),
    })
    decoded = spectrum_asn.decode("Spectrum-DAppControlData", encoded)
    key, payload = decoded["controlPayload"]
    assert key == "prbBlacklistControl"
    assert payload["blacklistedPRBs"] == [3, 7, 11]


# ----------------------------------------------------------------------
# JSON round-trip
# ----------------------------------------------------------------------

def test_json_sensing_policy_envelope_shape():
    """The wire JSON shape the dApp emits matches what the gNB decoder
    expects: {controlType, controlPayload: {sensingPolicyControl: {..}}}."""
    mask = [0x3F80] * 20
    envelope = {
        "controlType": "sensingPolicy",
        "controlPayload": {
            "sensingPolicyControl": {
                "maskPerSlot": mask,
                "deactivate": False,
            }
        },
    }
    blob = json.dumps(envelope).encode("utf-8")
    parsed = json.loads(blob.decode("utf-8"))
    assert parsed == envelope


def test_json_sensing_policy_passes_validator():
    """The Spectrum-DAppControlData JSON schema must accept both the
    sensingPolicy and prbBlacklist alternatives, with their inner payloads
    correctly wired in the schema's oneOf branch."""
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "spectrum", "defs",
        "e3sm_spectrum.json",
    )
    with open(schema_path) as f:
        root = json.load(f)
    schema = {**root["$defs"]["Spectrum-DAppControlData"], "$defs": root["$defs"]}
    envelope = {
        "controlType": "sensingPolicy",
        "controlPayload": {
            "sensingPolicyControl": {
                "maskPerSlot": [0x3F80] * 20,
                "deactivate": False,
            }
        },
    }
    # Should not raise.
    jsonschema.validate(envelope, schema)

    # The prbBlacklist alternative MUST also validate.
    prb = {
        "controlType": "prbBlacklist",
        "controlPayload": {
            "prbBlacklistControl": {"blacklistedPRBs": [3, 7, 11]},
        },
    }
    jsonschema.validate(prb, schema)
