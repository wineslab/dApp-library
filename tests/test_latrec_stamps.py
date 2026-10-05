"""Where the dApp stamps latrec stages, and which key each stamp carries.

The recorder is replaced by an in-memory log, so these hold whether or not the
installed libe3py was built with it."""
import os
import queue
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import spectrum.spectrum_dapp as sd  # noqa: E402
from e3interface import latrec  # noqa: E402
from e3interface.e3_interface import E3Interface  # noqa: E402
from e3interface.libe3_agent import EVENT_INDICATION, EVENT_XAPP_CONTROL  # noqa: E402
from spectrum.spectrum_dapp import SpectrumSharingDApp, Timings  # noqa: E402

pytest.importorskip("google.protobuf")

IDS = dict(DECODE_E3SM_BEGIN=0x1A, DECODE_E3SM_DONE=0x1B, ENCODE_E3SM_BEGIN=0x18,
           ENCODE_E3SM_DONE=0x19, PROCESS_DONE=0x40, CREATE_OUTPUT=0x41,
           APPLY_POLICY_DONE=0x42, PDU_INDICATION=1, PDU_XAPP_CONTROL=2,
           PDU_CONTROL=3, PDU_REPORT=4)


@pytest.fixture
def rec(monkeypatch):
    log, state = [], {"ctx": 0}
    for name, val in IDS.items():
        monkeypatch.setattr(latrec, name, val)
    monkeypatch.setattr(latrec, "ENABLED", True)
    monkeypatch.setattr(
        latrec, "stamp",
        lambda seq, stage, aux=0, aux2=0: log.append((seq, stage, int(aux), aux2)))
    monkeypatch.setattr(latrec, "ctx_set", lambda s: (log.append(("ctx_set", s)),
                                                       state.__setitem__("ctx", s)))
    monkeypatch.setattr(latrec, "ctx", lambda: state["ctx"])
    monkeypatch.setattr(latrec, "open_ring", lambda role: log.append(("ring", role)))
    return log, state


def _iface():
    iface = object.__new__(E3Interface)
    iface.indication_callbacks = {}
    iface.subscription_callbacks = {}
    iface.xapp_control_callbacks = {}
    iface._callback_lock = threading.Lock()
    iface.stop_event = threading.Event()
    iface.outbound_queue = queue.Queue()
    return iface


class _Ev:
    def __init__(self, kind, payload=b"x", request_id=0, trace_seq=0):
        self.kind, self.dapp_id, self.ran_function_id = kind, 1, 2
        self.request_id, self.trace_seq, self.sequence_id = request_id, trace_seq, 0
        self.subscription_id, self.response_code = 0, -1
        self._p = payload

    def get_payload(self):
        return self._p


class _InboundAgent:
    def __init__(self, iface, events):
        self.iface, self.events = iface, events

    def poll_events(self, max_batch, timeout_ms):
        batch, self.events = self.events, []
        if not batch:
            self.iface.stop_event.set()
        return batch

    def dropped_events(self):
        return 0


def test_inbound_thread_opens_a_ring_and_publishes_the_event_key(rec):
    log, state = rec
    iface = _iface()
    seen = []
    iface.add_indication_callback(1, 0, lambda d, rf, data: seen.append(("ind", state["ctx"])))
    iface.add_xapp_control_callback(1, 0, lambda d, data: seen.append(("ctl", state["ctx"])))
    iface.agent = _InboundAgent(iface, [
        _Ev(EVENT_INDICATION, trace_seq=11),
        _Ev(EVENT_XAPP_CONTROL, request_id=7, trace_seq=99),
        _Ev(EVENT_XAPP_CONTROL, request_id=0, trace_seq=5),
    ])
    iface._inbound_connection()
    assert log[0] == ("ring", "dapp.inbound")
    assert seen == [("ind", 11), ("ctl", 7), ("ctl", 5)]


def test_scheduled_messages_carry_the_producer_key_to_the_outbound_thread(rec):
    log, state = rec
    iface = _iface()
    state["ctx"] = 42
    iface.schedule_report(1, 1, b"rep")
    state["ctx"] = 43
    iface.schedule_control(1, 1, 1, b"ctl")
    iface.send_message_ack(8)
    state["ctx"] = 0
    log.clear()

    calls = []

    class Agent:
        def send_report(self, rf, data, seq_id=0):
            calls.append(("report", state["ctx"]))
            return 0

        def send_control(self, rf, cid, data, seq_id=0):
            calls.append(("control", state["ctx"]))
            return 0

        def send_message_ack(self, rid, positive):
            calls.append(("ack", state["ctx"]))
            return 0

    iface.agent = Agent()
    th = threading.Thread(target=iface._outbound_connection)
    th.start()
    while len(calls) < 3:
        threading.Event().wait(0.01)
    iface.stop_event.set()
    th.join(timeout=5)
    assert log[0] == ("ring", "dapp.outbound")
    assert calls == [("report", 42), ("control", 43), ("ack", 0)]


def _dapp(encoding="asn1"):
    d = object.__new__(SpectrumSharingDApp)
    d.encoding_method = encoding
    d.dapp_id = 7
    d._init_spectrum_encoder()
    return d


def test_indication_decode_is_bracketed_and_keys_the_worker_item(rec, monkeypatch):
    log, state = rec
    state["ctx"] = 11
    d = _dapp()
    got = []
    d._handle_l1_indication = lambda p, t: got.append(t)

    class _Ptr:
        @staticmethod
        def from_bytes(data, encoding=None):
            return object()
    monkeypatch.setattr(sd, "SlotPointer", _Ptr)
    d._handle_indication(7, SpectrumSharingDApp.RAN_FUNCTION_ID, b"abcd")
    assert log == [(11, 0x1A, 4, 1), (11, 0x1B, 1, 1)]
    assert got[0].lat_seq == 11


def test_worker_adopts_the_item_key_before_processing(rec):
    log, state = rec
    d = object.__new__(SpectrumSharingDApp)
    d._ind_worker_stop = threading.Event()
    d._ind_queue = queue.Queue()
    d._ind_worker_last_iter_ns = 0
    d._ind_worker_max_gap_ns = 0
    t = Timings(lat_seq=23)
    d._ind_queue.put(([], 0, 0, 0, (), 0x3FFF, t))
    seen = []
    d._process_indication = lambda *a: (seen.append(state["ctx"]), d._ind_worker_stop.set())
    d._indication_worker()
    assert log[0] == ("ring", "dapp.worker")
    assert seen == [23]


def test_report_encode_is_bracketed_by_create_and_encode_stamps(rec):
    log, state = rec
    state["ctx"] = 31
    wire = _dapp().create_prb_blacklist_report([3, 4])
    assert log == [(31, 0x41, 0, 4), (31, 0x18, 0, 4), (31, 0x19, len(wire), 4)]


def test_control_encode_uses_the_control_pdu_type(rec):
    log, state = rec
    state["ctx"] = 32
    wire = _dapp().create_prb_block_control([1, 2])
    assert [e[1] for e in log] == [0x41, 0x18, 0x19]
    assert log[2] == (32, 0x19, len(wire), 3)


def test_xapp_policy_is_decoded_then_applied_under_one_key(rec):
    log, state = rec
    state["ctx"] = 7
    d = _dapp()
    d.save_iqs = False
    applied = []
    d._reconcile_prb_blocks = lambda **kw: applied.append(kw)
    d._decode_xapp_control_envelope = lambda data: {
        "timestamp": None, "payload_key": "prbBlockedControl", "type": "prbBlocked",
        "payload": {"blockedPRBs": [5, 6, 7]}}
    d._handle_xapp_control(7, b"12345", 9)
    assert log == [(7, 0x1A, 5, 2), (7, 0x1B, 1, 2), (7, 0x42, 3, 0)]
    assert applied == [{"xapp": {5, 6, 7}, "sequence_id": 9}]


def test_unhandled_xapp_variant_is_decoded_but_not_applied(rec):
    log, state = rec
    state["ctx"] = 7
    d = _dapp()
    d._decode_xapp_control_envelope = lambda data: {
        "timestamp": None, "payload_key": "configControl", "type": "config", "payload": {}}
    d._handle_xapp_control(7, b"1", 0)
    assert [e[1] for e in log] == [0x1A, 0x1B]
