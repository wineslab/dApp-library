"""Round-trip tests for the E3 encoders.

Exercises every PDU type on both JsonE3Encoder (flat wire format, matches
libe3's json_encoder.cpp) and AsnE3Encoder. Also asserts the on-the-wire
shape for the JSON encoder: top-level ``type`` discriminator, no nested
``msg`` wrapper, and adaptive binary-payload handling (inline JSON for
structured bytes; ``{"__hex__": ...}`` sentinel for opaque bytes).
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from e3interface.e3_encoder import AsnE3Encoder, JsonE3Encoder  # noqa: E402


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def jenc():
    return JsonE3Encoder()


def _roundtrip(enc, pdu_type, build_kwargs, build_method):
    payload = getattr(enc, build_method)(**build_kwargs)
    decoded_type, body, msg_id = enc.decode_pdu(payload)
    assert decoded_type == pdu_type
    assert msg_id == build_kwargs["msgId"]
    return payload, body


def test_json_setup_request(jenc):
    payload, body = _roundtrip(
        jenc, "setupRequest",
        dict(msgId=42, e3apProtocolVersion="1.0.0",
             dAppName="SmokeApp", dAppVersion="0.1.0", vendor="WinesLab"),
        "create_setup_request",
    )
    assert body["dAppName"] == "SmokeApp"

    # Wire shape: flat, type-discriminated, no nested msg
    parsed = json.loads(payload.decode("utf-8"))
    assert parsed["type"] == "setupRequest"
    assert parsed["id"] == 42
    assert "msg" not in parsed
    assert parsed["dAppName"] == "SmokeApp"


def test_json_subscription_request_optional_fields(jenc):
    _, body = _roundtrip(
        jenc, "subscriptionRequest",
        dict(msgId=7, dappId=3, ranFunctionId=2,
             telemetryIds=[1, 4, 5, 6], controlIds=[], periodicity=100),
        "create_subscription_request",
    )
    assert body["telemetryIdentifierList"] == [1, 4, 5, 6]
    assert body["periodicity"] == 100
    assert "subscriptionTime" not in body


def test_json_subscription_delete(jenc):
    _roundtrip(
        jenc, "subscriptionDelete",
        dict(msgId=11, dappId=3, subscriptionId=1),
        "create_subscription_delete",
    )


def test_json_message_ack(jenc):
    _, body = _roundtrip(
        jenc, "messageAck",
        dict(msgId=99, requestId=42, responseCode="positive"),
        "create_message_ack",
    )
    assert body["responseCode"] == "positive"


def test_json_release(jenc):
    _roundtrip(
        jenc, "releaseMessage",
        dict(msgId=8, dappId=3),
        "create_release_message",
    )


def test_json_indication_inline_json(jenc):
    inner = {"sfn": 100, "slot": 7, "iqSamples": "aabbcc"}
    _, body = _roundtrip(
        jenc, "indicationMessage",
        dict(msgId=15, dappId=3, ranFunctionId=2,
             protocolData=json.dumps(inner).encode("utf-8")),
        "create_indication_message",
    )
    assert json.loads(body["protocolData"].decode("utf-8")) == inner


def test_json_indication_hex_sentinel_fallback(jenc):
    raw = b"\x00\x01\x02\xff"
    _, body = _roundtrip(
        jenc, "indicationMessage",
        dict(msgId=16, dappId=3, ranFunctionId=2, protocolData=raw),
        "create_indication_message",
    )
    assert body["protocolData"] == raw


def test_json_control_action(jenc):
    inner = {"blacklistedPRBs": [0, 1, 2]}
    _, body = _roundtrip(
        jenc, "dAppControlAction",
        dict(msgId=21, dappId=3, ranFunctionId=1, controlId=1,
             actionData=json.dumps(inner).encode("utf-8")),
        "create_control_action",
    )
    assert json.loads(body["actionData"].decode("utf-8")) == inner


def test_json_dapp_report(jenc):
    inner = {"foo": "bar"}
    _, body = _roundtrip(
        jenc, "dAppReport",
        dict(msgId=33, dappId=3, ranFunctionId=1,
             reportData=json.dumps(inner).encode("utf-8")),
        "create_dapp_report",
    )
    assert json.loads(body["reportData"].decode("utf-8")) == inner


def test_json_decode_libe3_setup_response(jenc):
    """Mimic the bytes libe3 puts on the wire for a setupResponse and verify
    the decoder converts each ranFunctionData back to bytes (matching libe3's
    1-byte 0x00 placeholder for empty data)."""
    wire = {
        "type": "setupResponse",
        "id": 42,
        "timestamp": 0,
        "requestId": 42,
        "responseCode": "positive",
        "ranIdentifier": "oai-gnb",
        "dAppIdentifier": 3,
        "ranFunctionList": [
            {"ranFunctionIdentifier": 1, "telemetryIdentifierList": [],
             "controlIdentifierList": [1], "ranFunctionData": {"__hex__": "00"}},
            {"ranFunctionIdentifier": 2, "telemetryIdentifierList": [1, 4, 5, 6],
             "controlIdentifierList": [], "ranFunctionData": {"__hex__": "00"}},
        ],
    }
    pdu_type, body, msg_id = jenc.decode_pdu(json.dumps(wire).encode("utf-8"))
    assert pdu_type == "setupResponse"
    assert msg_id == 42
    assert body["dAppIdentifier"] == 3
    for func in body["ranFunctionList"]:
        assert func["ranFunctionData"] == b"\x00"


# ---------------------------------------------------------------------------
# ASN.1 (regression — make sure my refactor didn't break the legacy path)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def aenc():
    return AsnE3Encoder()


def test_asn1_setup_request(aenc):
    _roundtrip(
        aenc, "setupRequest",
        dict(msgId=42, e3apProtocolVersion="1.0.0",
             dAppName="SmokeApp", dAppVersion="0.1.0", vendor="WinesLab"),
        "create_setup_request",
    )


def test_asn1_indication(aenc):
    _, body = _roundtrip(
        aenc, "indicationMessage",
        dict(msgId=15, dappId=3, ranFunctionId=2, protocolData=b"\x00\x01\x02"),
        "create_indication_message",
    )
    assert body["protocolData"] == b"\x00\x01\x02"
