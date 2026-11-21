"""Thin Python facade over the libe3 SWIG bindings (``libe3py``).

libe3 owns all E3AP operations (transport, setup handshake, subscribe,
indication/control framing, wire encoding). This module adapts
``libe3py.DAppSession`` — the batched, lock-free dApp seam — into an
ergonomic Python object that :class:`e3interface.e3_interface.E3Interface`
drives. Service-model (E3SM) payloads stay opaque here as ``bytes`` and are
encoded/decoded in the dApp subclasses (spectrum/simple), mirroring the OAI
split where libe3 handles E3AP and the SM handles encode/decode.

``libe3py`` comes from PyPI (a dependency of this package) and is built against
the libe3 installed on the machine; install libe3 once first::

    ./build_libe3 --all-encodings --install
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .e3_logging import e3_logger

# libe3py is imported lazily so that modules which only need the callback layer
# (e.g. tests exercising E3Interface dispatch) import fine without the bindings
# installed. Constructing a Libe3Agent requires libe3py.
try:
    import libe3py as _libe3
except ImportError:  # pragma: no cover - environment dependent
    _libe3 = None

LIBE3PY_AVAILABLE = _libe3 is not None

#: Minimum libe3 this package works against. 0.6.0 moves the POSIX link to one endpoint per
#: dApp (IPC path ``/tmp/dapps/e3``, one TCP or SCTP port) with new framing, so a dApp built
#: on an older binding cannot reach a gNB built on 0.6 over POSIX, nor the reverse; 0.5.0 had
#: already put a new PDU (E3-ControlResponse) in the E3AP choice; 0.6.1 stops exporting libe3's asn1c
#: runtime. Both ends must be the same release. ZMQ keeps its three endpoints and is unaffected
#: by the POSIX change.
LIBE3_MIN_VERSION = "0.6.1"


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split(".")[:3])


def _libe3py_is_current() -> bool:
    """Whether the imported binding is libe3 >= :data:`LIBE3_MIN_VERSION`.

    Bindings from libe3 0.2.2 on export ``__version__``; an older one exports none and is
    too old for any release this package accepts.
    """
    version = getattr(_libe3, "__version__", None)
    return version is not None and _version_tuple(version) >= _version_tuple(LIBE3_MIN_VERSION)


#: True when libe3py is importable *and* new enough to drive this package.
LIBE3PY_USABLE = LIBE3PY_AVAILABLE and _libe3py_is_current()

_BUILD_CMD = (
    "  # once, in a libe3 checkout of the same version (https://github.com/wineslab/libe3):\n"
    "  ./build_libe3 -I && ./build_libe3 --all-encodings --install\n"
    "  # then, in this environment:\n"
    "  pip install --force-reinstall --no-cache-dir libe3py"
)

_IMPORT_HINT = (
    "libe3py is not importable. Install libe3, then its Python binding into "
    "this environment:\n" + _BUILD_CMD
)

_VERSION_HINT = (
    f"the installed libe3py predates libe3 {LIBE3_MIN_VERSION}. 0.6.0 serves a dApp on one "
    "POSIX endpoint, so an older binding and the gNB's libe3 cannot talk over the POSIX link; "
    f"0.6.1 keeps libe3's asn1c runtime private. Install libe3 and libe3py of the 0.6 "
    f"series, {LIBE3_MIN_VERSION} or newer:\n" + _BUILD_CMD
)

# ErrorCode.SUCCESS is 0 in libe3.
SUCCESS = 0

# Inbound event kinds — stable integer ABI matching libe3
# swig/e3_dapp_session.hpp (E3EventKind). Re-exported so callers don't reach
# into libe3py, and so this module imports even when the bindings are absent.
EVENT_NONE = 0
EVENT_INDICATION = 1
EVENT_XAPP_CONTROL = 2
EVENT_SUBSCRIPTION_RESPONSE = 3
EVENT_SETUP_RESPONSE = 4
EVENT_MESSAGE_ACK = 5  # DEPRECATED TO BE REMOVED in libe3: it sends no ack, and reports one from another RAN
EVENT_CONTROL_RESPONSE = 7

# E3AP refusal cause the dApp acts on: the same value as libe3's Cause enum (types.hpp),
# a stable wire value. A telemetry or control id the RAN function does not advertise.
CAUSE_UNSUPPORTED_IDENTIFIER = 6


def _enum_maps():
    """Build the string->libe3py-enum maps (requires libe3py)."""
    return (
        {"zmq": _libe3.E3LinkLayer_ZMQ, "posix": _libe3.E3LinkLayer_POSIX},
        {"sctp": _libe3.E3TransportLayer_SCTP, "tcp": _libe3.E3TransportLayer_TCP,
         "ipc": _libe3.E3TransportLayer_IPC},
        {"asn1": _libe3.EncodingFormat_ASN1, "json": _libe3.EncodingFormat_JSON,
         "protobuf": _libe3.EncodingFormat_PROTOBUF},
    )


@dataclass
class RanFunctionEntry:
    """One RAN function advertised in the SetupResponse."""

    ran_function_identifier: int
    telemetry_identifier_list: list[int]
    control_identifier_list: list[int]
    ran_function_data: bytes

    def as_dict(self) -> dict:
        """Dict shape consumed by the dApp examples (see examples/*_dapp.py)."""
        return {
            "ranFunctionIdentifier": self.ran_function_identifier,
            "telemetryIdentifierList": list(self.telemetry_identifier_list),
            "controlIdentifierList": list(self.control_identifier_list),
            "ranFunctionData": self.ran_function_data,
        }


class Libe3Agent:
    """dApp-role E3AP agent backed by ``libe3py.DAppSession``."""

    def __init__(
        self,
        link: str = "zmq",
        transport: str = "ipc",
        encoding: str = "asn1",
        dapp_name: str = "dapp-library",
        dapp_version: str = "0.0.0",
        vendor: str = "",
        e3ap_version: str = "1.0.0",
        log_level: int = 3,
        queue_capacity: int = 8192,
    ):
        # NOTE: no ``**kwargs`` catch-all — an unknown keyword (e.g. an
        # E3Config field like ``setup_endpoint`` that is not wired here) must
        # fail loudly with a TypeError rather than be silently dropped and then
        # surface later as an opaque setup timeout.
        if not LIBE3PY_AVAILABLE:
            raise ImportError(_IMPORT_HINT)
        # Fail here, with the reason, rather than several frames deeper inside
        # libe3 at setup time where the message names nothing actionable.
        if not LIBE3PY_USABLE:
            raise ImportError(_VERSION_HINT)

        link = (link or "zmq").lower()
        transport = (transport or "ipc").lower()
        encoding = (encoding or "asn1").lower()
        link_map, transport_map, encoding_map = _enum_maps()
        if link not in link_map:
            raise ValueError(f"Unsupported link layer: {link!r}")
        if transport not in transport_map:
            raise ValueError(f"Unsupported transport layer: {transport!r}")
        if encoding not in encoding_map:
            raise ValueError(f"Unsupported encoding: {encoding!r}")

        cfg = _libe3.E3Config()
        cfg.role = _libe3.E3Role_DAPP
        cfg.link_layer = link_map[link]
        cfg.transport_layer = transport_map[transport]
        cfg.encoding = encoding_map[encoding]
        cfg.dapp_name = dapp_name
        cfg.dapp_version = dapp_version
        cfg.vendor = vendor
        cfg.e3ap_version = e3ap_version
        cfg.log_level = log_level

        self._cfg = cfg
        self._link = link
        self._transport = transport
        self._encoding = encoding
        self._session = _libe3.DAppSession(cfg, queue_capacity)
        e3_logger.info(
            "Libe3Agent created (link=%s transport=%s encoding=%s dapp=%s)",
            link, transport, encoding, dapp_name,
        )

    # --- lifecycle --------------------------------------------------------------

    def start(self) -> int:
        """Start the agent (returns libe3 ErrorCode; 0 == success)."""
        return self._session.start()

    def wait_for_setup(self, timeout_ms: int) -> int:
        """Block until the setup handshake completes (ErrorCode; 0 == success)."""
        return self._session.wait_for_setup(int(timeout_ms))

    def release(self) -> int:
        return self._session.release()

    def stop(self) -> None:
        self._session.stop()

    # --- setup introspection ----------------------------------------------------

    @property
    def dapp_id(self) -> Optional[int]:
        did = self._session.dapp_id()
        return None if did < 0 else int(did)

    @property
    def ran_identifier(self) -> str:
        return self._session.ran_identifier()

    def setup_positive(self) -> bool:
        return self._session.setup_response_code() == 0  # ResponseCode.POSITIVE

    def ran_function_list(self) -> list[RanFunctionEntry]:
        out: list[RanFunctionEntry] = []
        for i in range(self._session.setup_ran_function_count()):
            out.append(
                RanFunctionEntry(
                    ran_function_identifier=int(self._session.setup_ran_function_id(i)),
                    telemetry_identifier_list=list(self._session.setup_ran_function_telemetry(i)),
                    control_identifier_list=list(self._session.setup_ran_function_control(i)),
                    ran_function_data=bytes(self._session.setup_ran_function_data(i)),
                )
            )
        return out

    def setup_response_dict(self) -> dict:
        """Rebuild the SetupResponse dict shape the dApp base + examples expect."""
        return {
            "dAppIdentifier": self.dapp_id,
            "responseCode": "positive" if self.setup_positive() else "negative",
            "ranIdentifier": self.ran_identifier,
            "ranFunctionList": [e.as_dict() for e in self.ran_function_list()],
        }

    # --- outbound verbs ---------------------------------------------------------

    def subscribe(
        self,
        ran_function_id: int,
        telemetry_ids: list[int],
        control_ids: list[int],
        sub_time: Optional[int] = None,
        periodicity: Optional[int] = None,
    ) -> int:
        """Send a subscribe request.

        Returns the assigned request id (a positive int) on success —
        so the caller can correlate the SubscriptionResponse (which echoes
        ``request_id``) back to this call — or a negative libe3 ErrorCode on
        failure.
        """
        return self._session.subscribe(
            int(ran_function_id),
            _libe3.Uint32Vec(list(telemetry_ids)),
            _libe3.Uint32Vec(list(control_ids)),
            -1 if sub_time is None else int(sub_time),
            -1 if periodicity is None else int(periodicity),
        )

    def unsubscribe(self, ran_function_id: int) -> int:
        return self._session.unsubscribe(int(ran_function_id))

    def send_control(self, ran_function_id: int, control_id: int, action_data: bytes = b"",
                     sequence_id: int = 0, request_ack: bool = False) -> int:
        """Send a control to the RAN.

        ``sequence_id`` is the correlation id of the xApp procedure this control
        carries out. Echo the id from the xApp control that prompted it, so the
        RAN can report back when the decision reached the air; 0 means the dApp
        decided on its own and there is no procedure to answer.

        ``request_ack`` asks the RAN for an E3-ControlResponse once it has dealt
        with the control. The RAN answers a control it refuses itself either way.
        Returns the assigned message id, which is the ``request_id`` of the answer.
        """
        return self._session.send_control(int(ran_function_id), int(control_id), bytes(action_data),
                                          int(sequence_id), bool(request_ack))

    def send_control_response(self, request_id: int, ran_function_id: int, control_id: int,
                              positive: bool = True, cause: int = -1, detail: str = "",
                              outcome: bytes = b"") -> int:
        """Answer an xApp control whose ``request_ack`` is set.

        ``request_id`` is the event's ``request_id``. ``control_id`` is the control that
        was applied, from the list the RAN function advertises. ``cause`` is a libe3
        E3-Cause value (0 to 14), -1 for none; ``detail`` is text for it, up to 512 bytes;
        ``outcome`` is what the dApp reports about the control, up to 32768 bytes.
        """
        return self._session.send_control_response(int(request_id), int(ran_function_id),
                                                   int(control_id), 0 if positive else 1,
                                                   int(cause), str(detail), bytes(outcome))

    def send_report(self, ran_function_id: int, report_data: bytes, sequence_id: int) -> int:
        """Send a report to the RAN.

        ``sequence_id`` is the correlation id this dApp assigns to the detection;
        it travels to the xApp on the resulting E2 indication. It is mandatory on the
        wire (1..4294967295); 0 is not valid and is refused with a negative ErrorCode.
        """
        return self._session.send_report(int(ran_function_id), bytes(report_data), int(sequence_id))

    def send_message_ack(self, request_id: int, positive: bool = True) -> int:
        """DEPRECATED TO BE REMOVED in libe3. A libe3 RAN does not read it: answer an xApp
        control with :meth:`send_control_response`."""
        return self._session.send_message_ack(int(request_id), 0 if positive else 1)

    # --- inbound (batched drain) ------------------------------------------------

    def poll_events(self, max_batch: int, timeout_ms: int):
        """Drain up to ``max_batch`` inbound events, waiting ``timeout_ms`` for the first.

        Returns the SWIG ``E3EventVec`` (iterable of events, each with ``kind``,
        ``dapp_id``, ``ran_function_id``, ``subscription_id``, ``request_id``,
        ``sequence_id``, ``response_code``, ``control_id``, ``cause``, ``detail``,
        ``request_ack``, ``has_outcome`` and ``get_payload()`` returning native
        ``bytes``).
        Empty on timeout. The GIL is released for the whole call so libe3's
        threads run freely.
        """
        return self._session.poll_events(int(max_batch), int(timeout_ms))

    def dropped_events(self) -> int:
        return int(self._session.dropped_events())
