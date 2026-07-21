#!/usr/bin/env python3
"""
Minimal dApp for the TEST service model exposed by libe3's simple_agent example.

This dApp supports encoding/decoding of the simple service model defined in
`defs/e3sm_simple.asn`. Indications are decoded with `decode_indication()` and
control actions can be created with `create_control()`; encoding is performed
using ASN.1 PER by default (or JSON if `encoding_method="json"`).
"""

__author__ = "Andrea Lacava"

import time
import os
import json

from typing import override

import asn1tools
import jsonschema

from dapp.dapp import DApp
from e3interface.e3_logging import dapp_logger


class SimpleDApp(DApp):
    """Minimal dApp that pairs with the TestServiceModel in simple_agent.cpp."""

    # dApp metadata
    DAPP_NAME = "SimpleDApp"
    DAPP_VERSION = "1.0.0"
    VENDOR = "WinesLab"
    E3AP_PROTOCOL_VERSION = "1.0.0"

    # Must match the agent's TestServiceModel
    RAN_FUNCTION_ID = 1
    TELEMETRY_ID = [1]
    CONTROL_ID = [1]

    def __init__(
        self,
        dapp_name: str = DAPP_NAME,
        dapp_version: str = DAPP_VERSION,
        vendor: str = VENDOR,
        e3ap_protocol_version: str = E3AP_PROTOCOL_VERSION,
        link: str = "zmq",
        transport: str = "ipc",
        encoding_method: str = "asn1",
        control: bool = False,
        **kwargs,
    ):
        super().__init__(
            dapp_name=dapp_name,
            dapp_version=dapp_version,
            vendor=vendor,
            e3ap_protocol_version=e3ap_protocol_version,
            link=link,
            transport=transport,
            encoding_method=encoding_method,
            **kwargs,
        )
        self.control = control
        self.indication_count = 0

        # Initialize simple service model encoder/decoder
        self._init_simple_encoder()

    # ---- E3AP callbacks -------------------------------------------------- #

    @override
    def _handle_indication(self, dapp_identifier, data: bytes):
        """Process a raw indication payload from the TEST service model.

        The simple_agent sends 16 bytes:
          - bytes  0-3 : big-endian uint32 sequence number
          - bytes  4-15: pattern data (each byte = index + seq)
        """
        self.indication_count += 1

        # Structured decode via ASN.1/JSON (legacy raw format is not supported)
        try:
            msg = self.decode_indication(data)
            if not isinstance(msg, dict):
                dapp_logger.warning(
                    f"Decoded indication is not a dict: {type(msg)} from dApp {dapp_identifier}"
                )
                return

            seq = msg.get("data1")
            dapp_logger.info(
                f"[TEST] Decoded indication from dApp {dapp_identifier}: {msg}"
            )

            if seq is None:
                dapp_logger.error("Value seq passed by E3 Agent is None")
                return

            if self.control and seq % 5 == 0:
                self._do_control(seq)

            if seq % 3 == 0:
                report = {"bin1": seq}
                report_payload = self.create_report(report)
                self.e3_interface.schedule_report(dapp_identifier, self.RAN_FUNCTION_ID, report_payload)

        except Exception:
            dapp_logger.exception(
                "Failed to decode indication; ignoring"
            )
            return

    @override
    def _handle_xapp_control(self, dapp_identifier: int, data: bytes):
        try:

            msg = self.decode_config_control(data)
            dapp_logger.info(
                f"[TEST] Decoded Control Action from xApp: {msg}"
            )
    
        except Exception:
            dapp_logger.exception(
                "Failed to decode control action; ignoring"
            )
            return

    # ---- Encoding / decoding for Simple Service Model ----------------- #

    def _init_simple_encoder(self):
        """Initialize encoder/decoder for the simple ASN.1 service model.

        Supports `asn1` and `json` modes. In `asn1` mode the ASN.1
        specification in `defs/e3sm_simple.asn` is compiled with `asn1tools`.
        In `json` mode a JSON schema file `defs/e3sm_simple.json` is expected
        (optional) and payloads are encoded as UTF-8 JSON.
        """
        match getattr(self, "encoding_method", "asn1"):
            case "asn1":
                asn_file_path = os.path.join(
                    os.path.dirname(__file__), "defs", "e3sm_simple.asn"
                )
                self.simple_encoder = asn1tools.compile_files(asn_file_path, codec="per")
                dapp_logger.info("Simple ASN.1 encoder initialized")
            case "json":
                self.simple_encoder = "json"
                dapp_logger.info("Simple JSON encoder initialized")
            case _:
                raise ValueError(f"Unsupported encoding method: {self.encoding_method}")

    def _encode_simple_message(self, message_type: str, data: dict) -> bytes:
        """Encode a simple service-model message.

        message_type should match one of the ASN.1 type names in
        `e3sm_simple.asn` (for ASN.1 mode) or be a logical type for JSON mode.
        """
        if getattr(self, "encoding_method", "asn1") == "asn1":
            return self.simple_encoder.encode(message_type, data)
        elif getattr(self, "encoding_method", "asn1") == "json":
            # For the simple model we use plain JSON encoding
            return json.dumps(data).encode("utf-8")
        else:
            raise ValueError(f"Unsupported encoding method: {self.encoding_method}")

    def _decode_simple_message(self, message_type: str, data: bytes) -> dict:
        """Decode a simple service-model message payload.

        Returns a Python dict with the decoded fields.
        """
        if getattr(self, "encoding_method", "asn1") == "asn1":
            return self.simple_encoder.decode(message_type, data)
        elif getattr(self, "encoding_method", "asn1") == "json":
            try:
                return json.loads(data.decode("utf-8"))
            except Exception:
                dapp_logger.exception("Failed to decode JSON simple message")
                raise
        else:
            raise ValueError(f"Unsupported encoding method: {self.encoding_method}")

    @override
    def _decode_ran_function_data(self, data_bytes: bytes) -> dict | None:
        """Decode the SM Simple `ranFunctionData` attached to a SetupResponse entry.

        Returns the decoded dict or None on failure.
        """
        return self._decode_simple_message("Simple-RanFunctionData", data_bytes)

    # Envelope-aware wrappers ------------------------------------------- #
    # Each direction has a (Type, Payload) envelope; the inner type is
    # selected via the CHOICE arm. asn1tools represents CHOICE values as
    # ("alternative_name", value) tuples.

    def decode_indication(self, data: bytes) -> dict:
        """Decode a Simple-IndicationData envelope and return the inner
        Simple-Indication payload (the only variant today)."""
        env = self._decode_simple_message("Simple-IndicationData", data)
        if self.encoding_method == "asn1":
            payload_key, payload = env["indicationPayload"]
        else:
            payload_dict = env["indicationPayload"]
            payload_key, payload = next(iter(payload_dict.items()))
        if payload_key != "simpleIndication":
            raise ValueError(f"unexpected indication variant: {payload_key!r}")
        return payload

    def create_control(self, control_payload: dict) -> bytes:
        """Wrap a Simple-Control payload in the Simple-DAppControlData envelope."""
        if self.encoding_method == "asn1":
            envelope = {
                "controlType": "simple",
                "controlPayload": ("simpleControl", control_payload),
            }
        else:
            envelope = {
                "controlType": "simple",
                "controlPayload": {"simpleControl": control_payload},
            }
        return self._encode_simple_message("Simple-DAppControlData", envelope)

    def decode_config_control(self, data: bytes) -> dict:
        """Decode a Simple-XAppControlData envelope and return the inner
        Simple-ConfigControl payload."""
        env = self._decode_simple_message("Simple-XAppControlData", data)
        if self.encoding_method == "asn1":
            payload_key, payload = env["controlPayload"]
        else:
            payload_dict = env["controlPayload"]
            payload_key, payload = next(iter(payload_dict.items()))
        if payload_key != "configControl":
            raise ValueError(f"unexpected xApp control variant: {payload_key!r}")
        return payload

    def create_report(self, report_payload: dict) -> bytes:
        """Wrap a Simple-DAppReport payload in the Simple-DAppReportData envelope."""
        if self.encoding_method == "asn1":
            envelope = {
                "reportType": "simple",
                "reportPayload": ("simpleReport", report_payload),
            }
        else:
            envelope = {
                "reportType": "simple",
                "reportPayload": {"simpleReport": report_payload},
            }
        return self._encode_simple_message("Simple-DAppReportData", envelope)

    # ---- Control ------------------------------------------------ #

    def _do_control(self, seq: int):
        """Send a trivial control action back to the agent."""
        # Build a small control payload using the Simple-Control ASN.1 type.
        # ASN Simple-Control defines `samplingThreshold` (0..100) — map seq
        # into that range so it's a valid integer.
        try:
            sampling_threshold = int(seq) % 101
        except Exception:
            sampling_threshold = 0

        control_payload = {"samplingThreshold": sampling_threshold}
        try:
            action_data = self.create_control(control_payload)
            self.e3_interface.schedule_control(
                dappId=self.dapp_id,
                ranFunctionId=self.RAN_FUNCTION_ID,
                controlId=self.CONTROL_ID[0],
                actionData=action_data,
            )
            dapp_logger.info(
                f"[TEST] Sent Simple-Control with samplingThreshold={sampling_threshold} (seq #{seq})"
            )
        except Exception:
            dapp_logger.exception("Failed to send control action")

    # ---- DApp lifecycle -------------------------------------------------- #

    @override
    def _control_loop(self):
        """Main loop body — just sleeps; work happens in the indication callback."""
        time.sleep(1)

    @override
    def _stop(self):
        dapp_logger.info(
            f"SimpleDApp stopping. Total indications received: {self.indication_count}"
        )
