import os
import json
from abc import ABC, abstractmethod
import asn1tools
import jsonschema


class E3Encoder(ABC):
    """Abstract base class for E3 message encoding/decoding"""
    
    def __init__(self):
        self.encoding_type = "unknown"
    
    @abstractmethod
    def encode_pdu(self, pdu_type: str, pdu_data: dict) -> bytes:
        """Encode a PDU message to bytes"""
        pass
    
    @abstractmethod
    def decode_pdu(self, data: bytes) -> tuple:
        """Decode bytes to PDU message, returns (pdu_type, pdu_data, msg_id)"""
        pass
    
    @abstractmethod
    def create_setup_request(self, msgId: int = 1, e3apProtocolVersion: str = "0.0.0", dAppName: str = "", dAppVersion: str = "0.0.0", vendor: str = "") -> bytes:
        """Create a setup request message"""
        pass
    
    @abstractmethod
    def create_subscription_request(self, msgId: int = 1, dappId: int = 1, ranFunctionId: int = 1,
                                    telemetryIds: list[int] = [], controlIds: list[int] = [],
                                    subscriptionTime: int | None = None, periodicity: int | None = None) -> bytes:
        """Create a subscription request message"""
        pass

    @abstractmethod
    def create_subscription_delete(self, msgId: int, dappId: int, subscriptionId: int) -> bytes:
        """Create a subscription delete message"""
        pass
    
    @abstractmethod
    def create_message_ack(self, msgId: int, requestId: int, responseCode: str = "positive") -> bytes:
        """Create a message acknowledgment"""
        pass
    
    @abstractmethod
    def create_control_action(self, msgId: int, dappId: int = 1, ranFunctionId: int = 1,
                              controlId: int = 0, actionData: bytes = b"") -> bytes:
        """Create a dApp control action message"""
        pass
    
    @abstractmethod
    def create_dapp_report(self, msgId: int, dappId: int = 1, ranFunctionId: int = 1, reportData: bytes = b"") -> bytes:
        """Create a dApp report message"""
        pass
    
    @abstractmethod
    def create_indication_message(self, msgId: int, dappId: int = 1, ranFunctionId: int = 1, protocolData: bytes = b"") -> bytes:
        """Create an indication message"""
        pass

    @abstractmethod
    def create_release_message(self, msgId: int, dappId: int) -> bytes:
        """Create a release message"""
        pass


class AsnE3Encoder(E3Encoder):
    """ASN.1 implementation of E3 message encoding/decoding"""
    
    def __init__(self, asn_file_path: str = None):
        super().__init__()
        self.encoding_type = "asn1_per"
        
        # Load ASN.1 definitions
        if asn_file_path is None:
            asn_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "defs", "e3.asn")
        
        self.defs = asn1tools.compile_files(asn_file_path, codec="per")
    
    def encode_pdu(self, pdu_type: str, pdu_data: dict) -> bytes:
        """Encode a PDU message to bytes using ASN.1 PER"""
        msg_id = pdu_data.pop("id", None)
        if msg_id is None:
            raise ValueError("Missing required PDU message id")
        pdu_message = {
            "id": msg_id,
            "msg": (pdu_type, pdu_data)
        }
        return self.defs.encode("E3-PDU", pdu_message)
    
    def decode_pdu(self, data: bytes) -> tuple:
        """Decode bytes to PDU message using ASN.1 PER, returns (pdu_type, pdu_data, msg_id)"""
        decoded = self.defs.decode('E3-PDU', data)
        msg_id = decoded.get("id")
        msg_type, msg_data = decoded.get("msg")
        return (msg_type, msg_data, msg_id)
    
    def create_setup_request(self, msgId: int = 1, e3apProtocolVersion: str = "0.0.0", dAppName: str = "", dAppVersion: str = "0.0.0", vendor: str = "") -> bytes:
        """Create a setup request message"""
        return self.encode_pdu("setupRequest", {
            "id": msgId,
            "e3apProtocolVersion": e3apProtocolVersion,
            "dAppName": dAppName,
            "dAppVersion": dAppVersion,
            "vendor": vendor
        })

    def create_subscription_request(self, msgId: int = 1, dappId: int = 1, ranFunctionId: int = 1,
                                    telemetryIds: list[int] = [], controlIds: list[int] = [],
                                    subscriptionTime: int | None = None, periodicity: int | None = None) -> bytes:
        """Create a subscription request message"""
        data = {
            "id": msgId,
            "dAppIdentifier": dappId,
            "ranFunctionIdentifier": ranFunctionId,
            "telemetryIdentifierList": telemetryIds,
            "controlIdentifierList": controlIds
        }
        if subscriptionTime is not None:
            data["subscriptionTime"] = subscriptionTime
        if periodicity is not None:
            data["periodicity"] = periodicity
        return self.encode_pdu("subscriptionRequest", data)

    def create_subscription_delete(self, msgId: int, dappId: int, subscriptionId: int) -> bytes:
        """Create a subscription delete message"""
        return self.encode_pdu("subscriptionDelete", {
            "id": msgId,
            "dAppIdentifier": dappId,
            "subscriptionId": subscriptionId
        })
    
    def create_message_ack(self, msgId: int, requestId: int, responseCode: str = "positive") -> bytes:
        """Create a message acknowledgment"""
        return self.encode_pdu("messageAck", {
            "id": msgId,
            "requestId": requestId,
            "responseCode": responseCode
        })
    
    def create_control_action(self, msgId: int, dappId: int, ranFunctionId: int,
                              controlId: int, actionData: bytes = b"") -> bytes:
        """Create a dApp control action message"""
        return self.encode_pdu("dAppControlAction", {
            "id": msgId,
            "dAppIdentifier": dappId,
            "ranFunctionIdentifier": ranFunctionId,
            "controlIdentifier": controlId,
            "actionData": actionData
        })
    
    def create_indication_message(self, msgId: int, dappId: int = 1, ranFunctionId: int = 1, protocolData: bytes = b"") -> bytes:
        """Create an indication message"""
        return self.encode_pdu("indicationMessage", {
            "id": msgId,
            "dAppIdentifier": dappId,
            "ranFunctionIdentifier": ranFunctionId,
            "protocolData": protocolData
        })
    
    def create_dapp_report(self, msgId: int, dappId: int = 1, ranFunctionId: int = 1, reportData: bytes = b"") -> bytes:
        """Create a dApp report message"""
        return self.encode_pdu("dAppReport", {
            "id": msgId,
            "dAppIdentifier": dappId,
            "ranFunctionIdentifier": ranFunctionId,
            "reportData": reportData
        })

    def create_release_message(self, msgId: int, dappId: int) -> bytes:
        """Create a release message"""
        return self.encode_pdu("releaseMessage", {
            "id": msgId,
            "dAppIdentifier": dappId
        })


class JsonE3Encoder(E3Encoder):
    """JSON implementation of E3 message encoding/decoding using JSON Schema validation.

    Wire format matches libe3's json_encoder (src/encoder/json_encoder.cpp):
    a single flat object ``{"type": "<pduType>", "id": N, "timestamp": ..., ...fields}``.
    Binary payload fields (protocolData / actionData / reportData / xAppControlData /
    ranFunctionData) are carried as inline JSON objects/arrays when the underlying
    bytes parse as structured JSON; otherwise the bytes are wrapped in a
    ``{"__hex__": "<hex>"}`` sentinel so they survive a roundtrip.
    """

    # Top-level PDU types that carry an opaque binary payload field on the wire.
    _BINARY_PAYLOAD_FIELDS = {
        "indicationMessage": "protocolData",
        "dAppControlAction": "actionData",
        "dAppReport": "reportData",
        "xAppControlAction": "xAppControlData",
    }

    def __init__(self, json_schema_path: str = None):
        super().__init__()
        self.encoding_type = "json"

        # Load JSON Schema definitions
        if json_schema_path is None:
            json_schema_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "defs", "e3.json")

        with open(json_schema_path, 'r') as f:
            self.schema = json.load(f)

        # Build a jsonschema validator with a registry derived from the schema's $defs.
        #
        # The validator's "current resource" determines how `#/$defs/...` refs
        # resolve. If we pass `self.schema["$defs"]["E3-PDU"]` directly as the
        # validator schema, `#/$defs/E3-MessageID` resolves *within* that
        # subschema (which has no $defs) and explodes with PointerToNowhere.
        # Instead, point the validator at a $ref against our registered URI:
        # the registry then resolves the ref against the full schema (which
        # does have $defs), and downstream refs continue to work.
        self._validator_cls = jsonschema.validators.validator_for(self.schema)
        self._validator_cls.check_schema(self.schema)
        registry = self._build_registry()
        base_uri = self.schema.get("$id", "urn:e3-schema")
        self._pdu_validator = self._validator_cls(
            {"$ref": f"{base_uri}#/$defs/E3-PDU"},
            registry=registry,
        )

    def _build_registry(self):
        """
        Build a referencing.Registry from the top-level schema so that
        $ref resolution works without the removed RefResolver.
        Requires: jsonschema >= 4.18 (ships with referencing as a dependency).
        """
        import referencing
        import referencing.jsonschema
        resource = referencing.Resource.from_contents(
            self.schema,
            default_specification=referencing.jsonschema.DRAFT202012,
        )
        base_uri = self.schema.get("$id", "urn:e3-schema")
        return referencing.Registry().with_resource(base_uri, resource)

    # ------------------------------------------------------------------
    # Adaptive payload helpers (mirror libe3 json_encoder.cpp:bytes_to_json_payload)
    # ------------------------------------------------------------------

    @staticmethod
    def _bytes_to_payload(data):
        """Embed raw bytes as inline JSON if they parse as a structured value;
        otherwise wrap in the ``{"__hex__": ...}`` sentinel."""
        if data is None:
            return {}
        if not isinstance(data, (bytes, bytearray, memoryview)):
            # Already structured (dict/list/etc.) — pass through.
            return data
        b = bytes(data)
        if not b:
            return {}
        try:
            parsed = json.loads(b.decode("utf-8"))
            if isinstance(parsed, (dict, list)):
                return parsed
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        return {"__hex__": b.hex()}

    @staticmethod
    def _payload_to_bytes(payload) -> bytes:
        """Reverse of ``_bytes_to_payload``. Always returns bytes."""
        if payload is None:
            return b""
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return bytes(payload)
        if isinstance(payload, dict) and isinstance(payload.get("__hex__"), str):
            return bytes.fromhex(payload["__hex__"])
        if isinstance(payload, (dict, list)):
            return json.dumps(payload).encode("utf-8")
        if isinstance(payload, str):
            # Legacy: tolerate plain hex strings (older nested wire format).
            try:
                return bytes.fromhex(payload)
            except ValueError:
                return payload.encode("utf-8")
        return json.dumps(payload).encode("utf-8")

    # ------------------------------------------------------------------
    # Inner-SM binary-field helpers (used by spectrum_dapp.py for fields
    # like ``iqSamples`` / ``blockedPRBs`` inside the SM payload).
    # Not used by encode_pdu/decode_pdu; the outer PDU's binary fields use
    # the adaptive helpers above instead.
    # ------------------------------------------------------------------

    @staticmethod
    def bytes_to_hex(data: bytes) -> str:
        """Convert bytes to hex string for JSON encoding"""
        return data.hex()

    @staticmethod
    def hex_to_bytes(hex_str: str) -> bytes:
        """Convert hex string to bytes for JSON decoding"""
        return bytes.fromhex(hex_str)

    @classmethod
    def _convert_binary_fields(cls, message_type: str, data: dict,
                                converter, expected_types,
                                binary_fields: dict) -> dict:
        fields = binary_fields.get(message_type, [])
        for field in fields:
            if field in data and isinstance(data[field], expected_types):
                data[field] = converter(data[field])
        return data

    @classmethod
    def prepare_data_for_json_encode(cls, message_type: str, data: dict,
                                     binary_fields: dict) -> dict:
        """Convert bytes fields to hex strings for JSON encoding"""
        return cls._convert_binary_fields(
            message_type, data, cls.bytes_to_hex, (bytes, bytearray, memoryview),
            binary_fields=binary_fields
        )

    @classmethod
    def prepare_data_from_json_decode(cls, message_type: str, data: dict,
                                      binary_fields: dict) -> dict:
        """Convert hex string fields back to bytes after JSON decoding"""
        return cls._convert_binary_fields(
            message_type, data, cls.hex_to_bytes, str,
            binary_fields=binary_fields
        )

    # ------------------------------------------------------------------
    # Core encode / decode (flat wire format)
    # ------------------------------------------------------------------

    def _validate_pdu(self, pdu: dict) -> None:
        self._pdu_validator.validate(pdu)

    def encode_pdu(self, pdu_type: str, pdu_data: dict) -> bytes:
        """Encode a PDU message to JSON bytes (flat wire format).

        ``pdu_data`` is a flat dict of fields. ``id`` is required; ``timestamp``
        is optional and defaults to 0 (matches libe3's encode behavior for
        unset timestamps). Any binary payload field for the given pdu_type is
        translated through ``_bytes_to_payload``.
        """
        # Work on a shallow copy so we don't mutate the caller's dict.
        body = dict(pdu_data)
        msg_id = body.pop("id", None)
        if msg_id is None:
            raise ValueError("Missing required PDU message id")
        timestamp = body.pop("timestamp", 0)

        binary_field = self._BINARY_PAYLOAD_FIELDS.get(pdu_type)
        if binary_field and binary_field in body:
            body[binary_field] = self._bytes_to_payload(body[binary_field])

        pdu = {"type": pdu_type, "id": msg_id, "timestamp": timestamp}
        pdu.update(body)
        self._validate_pdu(pdu)
        return json.dumps(pdu).encode('utf-8')

    def decode_pdu(self, data: bytes) -> tuple:
        """Decode flat JSON bytes to PDU message.

        Returns ``(pdu_type, pdu_data, msg_id)`` where ``pdu_data`` contains
        the remaining fields with binary payloads converted back to ``bytes``.
        """
        pdu = json.loads(data.decode('utf-8'))
        self._validate_pdu(pdu)

        pdu_type = pdu.pop("type", None)
        if pdu_type is None:
            raise ValueError("PDU missing required 'type' discriminator")
        msg_id = pdu.pop("id", None)
        if msg_id is None:
            raise ValueError("PDU missing required 'id' field")
        pdu.pop("timestamp", None)

        binary_field = self._BINARY_PAYLOAD_FIELDS.get(pdu_type)
        if binary_field and binary_field in pdu:
            pdu[binary_field] = self._payload_to_bytes(pdu[binary_field])

        if pdu_type == "setupResponse":
            for func in pdu.get("ranFunctionList", []) or []:
                if "ranFunctionData" in func:
                    func["ranFunctionData"] = self._payload_to_bytes(func["ranFunctionData"])

        return (pdu_type, pdu, msg_id)

    # ------------------------------------------------------------------
    # Message factories
    # ------------------------------------------------------------------

    def create_setup_request(self, msgId: int = 1, e3apProtocolVersion: str = "0.0.0",
                             dAppName: str = "", dAppVersion: str = "0.0.0", vendor: str = "") -> bytes:
        """Create a setup request message"""
        return self.encode_pdu("setupRequest", {
            "id": msgId,
            "e3apProtocolVersion": e3apProtocolVersion,
            "dAppName": dAppName,
            "dAppVersion": dAppVersion,
            "vendor": vendor
        })

    def create_subscription_request(self, msgId: int = 1, dappId: int = 1, ranFunctionId: int = 1,
                                    telemetryIds: list[int] = [], controlIds: list[int] = [],
                                    subscriptionTime: int | None = None, periodicity: int | None = None) -> bytes:
        """Create a subscription request message"""
        data = {
            "id": msgId,
            "dAppIdentifier": dappId,
            "ranFunctionIdentifier": ranFunctionId,
            "telemetryIdentifierList": telemetryIds,
            "controlIdentifierList": controlIds
        }
        if subscriptionTime is not None:
            data["subscriptionTime"] = subscriptionTime
        if periodicity is not None:
            data["periodicity"] = periodicity
        return self.encode_pdu("subscriptionRequest", data)

    def create_subscription_delete(self, msgId: int, dappId: int, subscriptionId: int) -> bytes:
        """Create a subscription delete message"""
        return self.encode_pdu("subscriptionDelete", {
            "id": msgId,
            "dAppIdentifier": dappId,
            "subscriptionId": subscriptionId
        })

    def create_message_ack(self, msgId: int, requestId: int, responseCode: str = "positive") -> bytes:
        """Create a message acknowledgment"""
        return self.encode_pdu("messageAck", {
            "id": msgId,
            "requestId": requestId,
            "responseCode": responseCode
        })

    def create_control_action(self, msgId: int, dappId: int, ranFunctionId: int,
                              controlId: int, actionData: bytes = b"") -> bytes:
        """Create a dApp control action message"""
        return self.encode_pdu("dAppControlAction", {
            "id": msgId,
            "dAppIdentifier": dappId,
            "ranFunctionIdentifier": ranFunctionId,
            "controlIdentifier": controlId,
            "actionData": actionData,
        })

    def create_indication_message(self, msgId: int, dappId: int = 1, ranFunctionId: int = 1,
                                  protocolData: bytes = b"") -> bytes:
        """Create an indication message"""
        return self.encode_pdu("indicationMessage", {
            "id": msgId,
            "dAppIdentifier": dappId,
            "ranFunctionIdentifier": ranFunctionId,
            "protocolData": protocolData,
        })

    def create_dapp_report(self, msgId: int, dappId: int = 1, ranFunctionId: int = 1,
                           reportData: bytes = b"") -> bytes:
        """Create a dApp report message"""
        return self.encode_pdu("dAppReport", {
            "id": msgId,
            "dAppIdentifier": dappId,
            "ranFunctionIdentifier": ranFunctionId,
            "reportData": reportData,
        })

    def create_release_message(self, msgId: int, dappId: int) -> bytes:
        """Create a release message"""
        return self.encode_pdu("releaseMessage", {
            "id": msgId,
            "dAppIdentifier": dappId
        })