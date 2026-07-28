#!/usr/bin/env python3
"""Minimal dApp for the Simple service model — the Python peer of libe3's
``examples/simple_agent.cpp`` (and mirror of ``examples/simple_dapp.cpp``).

It is wire-compatible with the C++ Simple SM: it reuses libe3's **installed**
``e3sm_simple.asn`` grammar (no local copy) so the exact same messages cross the
link. E3AP is handled entirely by libe3 (via ``libe3py``); this class only
encodes/decodes the Simple SM payloads.

Behaviour matches ``simple_dapp.cpp``: subscribe to RAN function 1 (telemetry
{1}, control {1}); on every 5th indication send a Simple-Control with
``samplingThreshold = seq % 101`` (when ``--control`` is set).

Both ASN.1 (default) and JSON encodings interoperate — libe3's Simple SM uses
bare, camelCase messages in both (``{"data1":..,"timestamp":..}`` etc.).
"""

__author__ = "Andrea Lacava"

import json
import os
import subprocess
import time
from typing import override

import asn1tools

from dapp.dapp import DApp
from e3interface.e3_logging import dapp_logger


def _locate_simple_asn() -> str:
    """Find the installed libe3 Simple SM grammar (e3sm_simple.asn).

    Resolution order: ``LIBE3_SM_SIMPLE_ASN`` env override, then
    ``pkg-config --variable=smdir libe3``, then the common install prefixes.
    """
    override = os.environ.get("LIBE3_SM_SIMPLE_ASN")
    if override and os.path.isfile(override):
        return override

    candidates: list[str] = []
    try:
        smdir = subprocess.check_output(
            ["pkg-config", "--variable=smdir", "libe3"], text=True
        ).strip()
        if smdir:
            candidates.append(os.path.join(smdir, "sm_simple", "e3sm_simple.asn"))
    except Exception:
        pass
    for prefix in ("/usr/local/share", "/usr/share"):
        candidates.append(os.path.join(prefix, "libe3", "sm", "sm_simple", "e3sm_simple.asn"))

    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Could not locate the installed libe3 e3sm_simple.asn. Install libe3 "
        "(./build_libe3 --install) or set LIBE3_SM_SIMPLE_ASN to its path. "
        f"Tried: {candidates}"
    )


class SimpleDApp(DApp):
    """Minimal dApp that pairs with the SimpleServiceModel in simple_agent.cpp."""

    DAPP_NAME = "SimpleDApp"
    DAPP_VERSION = "1.0.0"
    VENDOR = "WinesLab"
    E3AP_PROTOCOL_VERSION = "1.0.0"

    # Must match the agent's SimpleServiceModel (RAN function 1).
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
        self._init_simple_encoder()

    # ---- E3AP callbacks -------------------------------------------------- #

    @override
    def _handle_indication(self, dapp_identifier, ran_function_id, data: bytes):
        """Decode a Simple-Indication and, if enabled, echo a control every 5th."""
        self.indication_count += 1
        try:
            msg = self._decode_simple_message("Simple-Indication", data)
            seq = msg.get("data1")
            dapp_logger.info(f"[SIMPLE] indication from dApp {dapp_identifier}: {msg}")
            if seq is None:
                dapp_logger.error("Simple-Indication missing data1")
                return
            if self.control and seq % 5 == 0:
                self._do_control(seq)
        except Exception:
            dapp_logger.exception("Failed to decode Simple-Indication; ignoring")

    @override
    def _handle_xapp_control(self, dapp_identifier: int, data: bytes):
        try:
            msg = self._decode_simple_message("Simple-ConfigControl", data)
            dapp_logger.info(f"[SIMPLE] xApp ConfigControl: {msg}")
        except Exception:
            dapp_logger.exception("Failed to decode Simple-ConfigControl; ignoring")

    # ---- Simple SM encode/decode (bare messages, reuses libe3 grammar) --- #

    # ASN.1 message-type name -> protobuf message class in defs/e3sm_simple_pb2
    # (the .proto is vendored from libe3 examples/sm_simple to stay wire-
    # compatible with libe3's simple_agent protobuf path).
    _SIMPLE_PB_TYPES = {
        "Simple-Indication": "SimpleIndication",
        "Simple-Control": "SimpleControl",
        "Simple-ConfigControl": "SimpleConfigControl",
        "Simple-DAppReport": "SimpleDAppReport",
        "Simple-RanFunctionData": "SimpleRanFunctionData",
    }

    def _init_simple_encoder(self):
        """Compile the installed libe3 Simple SM ASN.1 grammar (or use JSON/protobuf)."""
        match self.encoding_method:
            case "asn1":
                asn_path = _locate_simple_asn()
                self.simple_encoder = asn1tools.compile_files(asn_path, codec="per")
                dapp_logger.info("Simple ASN.1 encoder initialized from %s", asn_path)
            case "json":
                self.simple_encoder = "json"
                dapp_logger.info("Simple JSON encoder initialized")
            case "protobuf":
                from .defs import e3sm_simple_pb2 as _simple_pb2
                self._simple_pb2 = _simple_pb2
                self.simple_encoder = "protobuf"
                dapp_logger.info("Simple protobuf encoder initialized")
            case _:
                raise ValueError(f"Unsupported encoding method: {self.encoding_method}")

    def _simple_pb_new(self, message_type: str):
        return getattr(self._simple_pb2, self._SIMPLE_PB_TYPES[message_type])()

    def _encode_simple_message(self, message_type: str, data: dict) -> bytes:
        if self.encoding_method == "protobuf":
            from google.protobuf import json_format
            msg = self._simple_pb_new(message_type)
            json_format.ParseDict(data, msg)
            return msg.SerializeToString()
        if self.encoding_method == "asn1":
            return self.simple_encoder.encode(message_type, data)
        # JSON: bare camelCase object, matching libe3's sm_simple wrapper.
        return json.dumps(data).encode("utf-8")

    def _decode_simple_message(self, message_type: str, data: bytes) -> dict:
        if self.encoding_method == "protobuf":
            from e3interface import sm_helpers
            msg = self._simple_pb_new(message_type)
            msg.ParseFromString(bytes(data))
            return sm_helpers.pb_message_to_dict(msg)
        if self.encoding_method == "asn1":
            return self.simple_encoder.decode(message_type, data)
        return json.loads(bytes(data).decode("utf-8"))

    @override
    def _decode_ran_function_data(self, data_bytes: bytes) -> dict | None:
        return self._decode_simple_message("Simple-RanFunctionData", data_bytes)

    # ---- Control -------------------------------------------------------- #

    def _do_control(self, seq: int):
        """Send a Simple-Control(samplingThreshold=seq%101), like simple_dapp.cpp."""
        sampling_threshold = int(seq) % 101
        try:
            action_data = self._encode_simple_message(
                "Simple-Control", {"samplingThreshold": sampling_threshold})
            self.e3_interface.schedule_control(
                dappId=self.dapp_id,
                ranFunctionId=self.RAN_FUNCTION_ID,
                controlId=self.CONTROL_ID[0],
                actionData=action_data,
            )
            dapp_logger.info(
                f"[SIMPLE] sent Simple-Control samplingThreshold={sampling_threshold} (seq #{seq})")
        except Exception:
            dapp_logger.exception("Failed to send Simple-Control")

    # ---- DApp lifecycle ------------------------------------------------- #

    @override
    def _control_loop(self):
        time.sleep(1)

    @override
    def _stop(self):
        dapp_logger.info(
            f"SimpleDApp stopping. Total indications received: {self.indication_count}")
