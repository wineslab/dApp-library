"""Service-model (E3SM) payload helpers — Python side of the E3AP/E3SM split.

These helpers operate on the *inner* SM payload fields (e.g. ``iqSamples`` /
``blockedPRBs`` inside a Spectrum SM message), converting binary fields to/from
hex strings for JSON SM encoding. They are E3SM concerns and stay in Python;
the outer E3AP PDU is handled entirely by libe3.

Previously these lived as static methods on ``JsonE3Encoder`` in the (now
removed) pure-Python E3AP layer; they were relocated here when libe3 took over
E3AP so the SM code keeps working unchanged.
"""

from __future__ import annotations


def bytes_to_hex(data: bytes) -> str:
    """Convert bytes to a hex string for JSON encoding."""
    return data.hex()


def hex_to_bytes(hex_str: str) -> bytes:
    """Convert a hex string back to bytes after JSON decoding."""
    return bytes.fromhex(hex_str)


def _convert_binary_fields(message_type: str, data: dict, converter,
                           expected_types, binary_fields: dict) -> dict:
    fields = binary_fields.get(message_type, [])
    for field in fields:
        if field in data and isinstance(data[field], expected_types):
            data[field] = converter(data[field])
    return data


def prepare_data_for_json_encode(message_type: str, data: dict, binary_fields: dict) -> dict:
    """Convert bytes fields to hex strings for JSON encoding."""
    return _convert_binary_fields(
        message_type, data, bytes_to_hex, (bytes, bytearray, memoryview),
        binary_fields=binary_fields,
    )


def prepare_data_from_json_decode(message_type: str, data: dict, binary_fields: dict) -> dict:
    """Convert hex string fields back to bytes after JSON decoding."""
    return _convert_binary_fields(
        message_type, data, hex_to_bytes, str,
        binary_fields=binary_fields,
    )


def pb_message_to_dict(msg) -> dict:
    """Convert a protobuf message to a native-typed dict for the SM decode path.

    Keys are each field's ``json_name`` — which the SM .proto files pin to the
    ASN.1/JSON field spellings (e.g. ``blacklistedPRBs``), so the result matches
    what the ASN.1 decode path returns and the rest of the dApp already consumes.

    Unlike ``google.protobuf.json_format.MessageToDict`` this keeps ``bytes`` as
    ``bytes`` (not base64), ``int64`` as ``int`` (not str), and includes proto3
    scalar defaults; ``optional``/oneof/message fields absent on the wire are
    omitted (checked via ``HasField``), matching ASN.1 OPTIONAL semantics.
    """
    from google.protobuf.descriptor import FieldDescriptor

    out: dict = {}
    for f in msg.DESCRIPTOR.fields:
        # is_repeated (protobuf >=5) avoids the deprecated .label accessor;
        # fall back to LABEL_REPEATED on 4.21.x.
        try:
            repeated = f.is_repeated
        except AttributeError:  # pragma: no cover - older protobuf
            repeated = f.label == FieldDescriptor.LABEL_REPEATED
        if not repeated and f.has_presence:
            # Explicit-presence field (proto3 `optional`, oneof member, or a
            # singular sub-message): only emit it when actually set.
            if not msg.HasField(f.name):
                continue
        value = getattr(msg, f.name)
        if repeated:
            out[f.json_name] = list(value)
        elif f.type == FieldDescriptor.TYPE_MESSAGE:
            out[f.json_name] = pb_message_to_dict(value)
        else:
            out[f.json_name] = value
    return out
