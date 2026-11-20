"""A control is answered with an E3-ControlResponse (libe3 0.5), and an xApp control that asks is
answered by this dApp.

libe3py is not needed: the agent facade is driven through a fake session and the dispatch loop through
a fake agent, as in the other dispatch tests.
"""
import logging
import os
import queue
import sys
import threading
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from e3interface import e3_interface, libe3_agent
from e3interface.e3_interface import E3Interface
from e3interface.libe3_agent import (
    EVENT_CONTROL_RESPONSE,
    EVENT_XAPP_CONTROL,
    Libe3Agent,
)

UNSUPPORTED_RAN_FUNCTION = 5
UNSPECIFIED = 0


class _Ev:
    """The fields of a libe3py E3Event that the dispatch reads."""

    def __init__(self, kind, **fields):
        self.kind = kind
        self.dapp_id, self.ran_function_id, self.control_id = 1, 2, 0
        self.request_id, self.sequence_id, self.trace_seq = 0, 0, 0
        self.request_ack, self.response_code, self.cause = False, -1, -1
        self.detail, self.has_outcome, self.payload = "", False, b"payload"
        self.__dict__.update(fields)

    def get_payload(self):
        return self.payload


class _InboundAgent:
    """Hands one batch to the inbound loop, then stops it."""

    def __init__(self, iface, events, control_ids=(3, 4)):
        self.iface, self.events, self.control_ids = iface, list(events), list(control_ids)

    def poll_events(self, max_batch, timeout_ms):
        batch, self.events = self.events, []
        if not batch:
            self.iface.stop_event.set()
        return batch

    def dropped_events(self):
        return 0

    def ran_function_list(self):
        return [types.SimpleNamespace(ran_function_identifier=2,
                                      control_identifier_list=self.control_ids)]


def _iface(agent_factory=None):
    iface = object.__new__(E3Interface)
    iface.indication_callbacks = {}
    iface.subscription_callbacks = {}
    iface.xapp_control_callbacks = {}
    iface.control_response_callbacks = {}
    iface._callback_lock = threading.Lock()
    iface.stop_event = threading.Event()
    iface.outbound_queue = queue.Queue()
    return iface


def _run_inbound(iface, events, **agent_kwargs):
    iface.agent = _InboundAgent(iface, events, **agent_kwargs)
    iface._inbound_connection()


def _queued(iface):
    """What the dApp queued for libe3, without the latency-recorder key."""
    out = []
    while not iface.outbound_queue.empty():
        kind, data = iface.outbound_queue.get()
        out.append((kind, {k: v for k, v in data.items() if k != "latSeq"}))
    return out


# ---- the agent facade ------------------------------------------------------------------------

class _Session:
    def __init__(self):
        self.calls = []

    def send_control(self, *args):
        self.calls.append(("send_control",) + args)
        return 11

    def send_control_response(self, *args):
        self.calls.append(("send_control_response",) + args)
        return 0


def _agent():
    agent = object.__new__(Libe3Agent)
    agent._session = _Session()
    return agent


def test_send_control_passes_request_ack_and_defaults_to_false():
    agent = _agent()
    assert agent.send_control(1, 2, b"x", 9, request_ack=True) == 11
    assert agent.send_control(1, 2, b"x") == 11
    assert agent._session.calls == [
        ("send_control", 1, 2, b"x", 9, True),
        ("send_control", 1, 2, b"x", 0, False),
    ]


def test_send_control_response_maps_its_arguments():
    agent = _agent()
    agent.send_control_response(7, 2, 3, positive=False, cause=7, detail="no", outcome=b"o")
    agent.send_control_response(8, 2, 3)
    assert agent._session.calls == [
        ("send_control_response", 7, 2, 3, 1, 7, "no", b"o"),
        ("send_control_response", 8, 2, 3, 0, -1, "", b""),
    ]


def test_the_control_response_event_kind_is_7():
    assert EVENT_CONTROL_RESPONSE == 7


@pytest.mark.parametrize("version,usable", [("0.4.3", False), ("0.5.0", True), ("0.5.2", True),
                                            ("0.10.0", True)])
def test_the_binding_must_be_libe3_0_5_or_newer(monkeypatch, version, usable):
    monkeypatch.setattr(libe3_agent, "_libe3", types.SimpleNamespace(__version__=version))
    assert libe3_agent._libe3py_is_current() is usable


def test_an_unversioned_binding_is_probed_for_request_ack(monkeypatch):
    new = types.SimpleNamespace(E3Event=types.SimpleNamespace(request_ack=False))
    old = types.SimpleNamespace(E3Event=types.SimpleNamespace(sequence_id=0))
    monkeypatch.setattr(libe3_agent, "_libe3", new)
    assert libe3_agent._libe3py_is_current() is True
    monkeypatch.setattr(libe3_agent, "_libe3", old)
    assert libe3_agent._libe3py_is_current() is False


# ---- a control that asks ---------------------------------------------------------------------

def test_schedule_control_carries_request_ack():
    iface = _iface()
    iface.schedule_control(1, 2, 3, b"a", requestAck=True)
    iface.schedule_control(1, 2, 3, b"b")
    (_, first), (_, second) = _queued(iface)
    assert first["requestAck"] is True
    assert second.get("requestAck", False) is False


def test_the_outbound_thread_sends_request_ack_to_libe3():
    iface = _iface()
    sent = []

    class Agent:
        def send_control(self, rf, cid, data, seq_id=0, request_ack=False):
            sent.append((rf, cid, data, seq_id, request_ack))
            return 5

    iface.agent = Agent()
    iface.schedule_control(1, 2, 3, b"a", sequenceId=4, requestAck=True)
    iface.schedule_control(1, 2, 3, b"b")
    t = threading.Thread(target=iface._outbound_connection)
    t.start()
    while not iface.outbound_queue.empty():
        pass
    iface.stop_event.set()
    t.join(timeout=5)
    assert sent == [(2, 3, b"a", 4, True), (2, 3, b"b", 0, False)]


def test_a_control_response_reaches_the_registered_callback_with_its_fields():
    iface = _iface()
    seen = []
    iface.add_control_response_callback(1, lambda d, resp: seen.append((d, resp)))
    _run_inbound(iface, [
        _Ev(EVENT_CONTROL_RESPONSE, request_id=12, ran_function_id=2, control_id=3,
            response_code=1, cause=7, detail="the scheduler refused it", has_outcome=True,
            payload=b"{}"),
    ])
    assert seen == [(1, {
        "requestId": 12, "ranFunctionId": 2, "controlId": 3, "responseCode": "negative",
        "cause": 7, "detail": "the scheduler refused it", "outcome": b"{}",
    })]


def test_a_positive_response_has_no_cause_and_no_outcome_unless_given():
    iface = _iface()
    seen = []
    iface.add_control_response_callback(1, lambda d, resp: seen.append(resp))
    _run_inbound(iface, [_Ev(EVENT_CONTROL_RESPONSE, request_id=5, response_code=0, control_id=3)])
    assert seen[0]["responseCode"] == "positive"
    assert seen[0]["cause"] is None
    assert seen[0]["detail"] == ""
    assert seen[0]["outcome"] is None


def test_a_negative_response_is_logged_even_with_no_callback(caplog):
    iface = _iface()
    with caplog.at_level(logging.WARNING, logger=e3_interface.e3_logger.name):
        _run_inbound(iface, [_Ev(EVENT_CONTROL_RESPONSE, request_id=5, response_code=1, cause=8,
                                 detail="no subscription to this RAN function")])
    assert any("no subscription to this RAN function" in r.getMessage() for r in caplog.records)


# ---- an xApp control that asks ---------------------------------------------------------------

def test_an_xapp_control_that_asks_is_answered_positive_after_the_callback_ran():
    iface = _iface()
    order = []
    iface.add_xapp_control_callback(1, 0, lambda d, data, seq: order.append("callback"))
    _run_inbound(iface, [_Ev(EVENT_XAPP_CONTROL, request_id=41, request_ack=True, sequence_id=9)])
    answers = _queued(iface)
    assert order == ["callback"]
    assert answers == [("control_response", {
        "requestId": 41, "ranFunctionId": 2, "controlId": 3, "positive": True,
        "cause": -1, "detail": "", "outcome": b"",
    })]


def test_an_xapp_control_that_does_not_ask_is_not_answered():
    iface = _iface()
    iface.add_xapp_control_callback(1, 0, lambda d, data, seq: None)
    _run_inbound(iface, [_Ev(EVENT_XAPP_CONTROL, request_id=41, request_ack=False)])
    assert _queued(iface) == []


def test_with_no_callback_an_xapp_control_that_asks_is_answered_negative():
    iface = _iface()
    _run_inbound(iface, [_Ev(EVENT_XAPP_CONTROL, request_id=41, request_ack=True)])
    [(kind, data)] = _queued(iface)
    assert kind == "control_response"
    assert data["positive"] is False
    assert data["cause"] == UNSUPPORTED_RAN_FUNCTION
    assert data["requestId"] == 41


def test_a_callback_that_raises_gets_a_negative_answer_and_the_loop_goes_on(caplog):
    iface = _iface()

    def broken(dapp_id, data, seq):
        raise RuntimeError("cannot apply")

    seen = []
    iface.add_xapp_control_callback(1, 0, broken)
    iface.add_indication_callback(1, 0, lambda d, rf, data: seen.append("indication"))
    from e3interface.libe3_agent import EVENT_INDICATION
    with caplog.at_level(logging.ERROR, logger=e3_interface.e3_logger.name):
        _run_inbound(iface, [_Ev(EVENT_XAPP_CONTROL, request_id=41, request_ack=True),
                             _Ev(EVENT_INDICATION)])
    [(kind, data)] = _queued(iface)
    assert (kind, data["positive"], data["cause"]) == ("control_response", False, UNSPECIFIED)
    assert "cannot apply" in data["detail"]
    assert seen == ["indication"]


def test_the_answer_names_a_control_the_ran_function_advertises():
    iface = _iface()
    iface.add_xapp_control_callback(1, 0, lambda d, data, seq: None)
    _run_inbound(iface, [_Ev(EVENT_XAPP_CONTROL, request_id=1, request_ack=True)], control_ids=(9,))
    [(_, data)] = _queued(iface)
    assert data["controlId"] == 9


def test_the_outbound_thread_sends_the_answer_through_libe3():
    iface = _iface()
    sent = []

    class Agent:
        def send_control_response(self, rid, rf, cid, positive, cause, detail, outcome):
            sent.append((rid, rf, cid, positive, cause, detail, outcome))
            return 0

    iface.agent = Agent()
    iface.send_control_response(41, 2, 3, positive=False, cause=7, detail="bad", outcome=b"o")
    t = threading.Thread(target=iface._outbound_connection)
    t.start()
    while not iface.outbound_queue.empty():
        pass
    iface.stop_event.set()
    t.join(timeout=5)
    assert sent == [(41, 2, 3, False, 7, "bad", b"o")]
