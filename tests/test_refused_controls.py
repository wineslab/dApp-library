#!/usr/bin/env python3
"""A control the RAN does not offer is disabled and the dApp subscribes again without it.

A libe3 RAN refuses a subscription that names a control id its RAN function does not
advertise. The refusal reaches the dApp with libe3's cause and text; the dApp drops those
controls, keeps the telemetry, and never sends them.
"""
import os
import queue
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from e3interface.e3_interface import E3Interface
from e3interface.libe3_agent import CAUSE_UNSUPPORTED_IDENTIFIER
from spectrum.spectrum_dapp import SpectrumSharingDApp


def _iface():
    """An E3Interface with only the subscription and outbound state."""
    iface = object.__new__(E3Interface)
    iface.subscription_callbacks = {}
    iface._callback_lock = threading.Lock()
    iface._sub_cv = threading.Condition()
    iface._sub_reqid_to_rf = {}
    iface._sub_pending_rfs = set()
    iface._sub_results = {}
    iface.outbound_queue = queue.Queue()
    iface.stop_event = threading.Event()
    iface.stop_event.set()  # keep __del__ from touching connections on GC
    return iface


def _event(response_code=1, cause=-1, detail="", request_id=42):
    return SimpleNamespace(dapp_id=7, response_code=response_code, subscription_id=0,
                           request_id=request_id, periodicity=-1, cause=cause, detail=detail)


# --- what the dApp is told ------------------------------------------------------

@pytest.mark.parametrize("cause, expected", [(-1, None), (6, 6), (5, 5)])
def test_cause_is_none_only_when_the_ran_gave_none(cause, expected):
    assert E3Interface._subscription_response_data(_event(cause=cause))["cause"] == expected


def test_detail_is_the_rans_text_or_none():
    assert E3Interface._subscription_response_data(_event(detail="unknown control id 1"))["detail"] \
        == "unknown control id 1"
    assert E3Interface._subscription_response_data(_event(detail=""))["detail"] is None


def test_an_older_libe3py_without_cause_reads_as_none():
    old = SimpleNamespace(dapp_id=7, response_code=1, subscription_id=0, request_id=42, periodicity=-1)
    data = E3Interface._subscription_response_data(old)
    assert data["cause"] is None and data["detail"] is None


def test_a_refusal_is_kept_per_ran_function_until_the_next_subscription():
    iface = _iface()
    iface._sub_pending_rfs.add(2)
    iface._sub_reqid_to_rf[42] = 2
    iface._handle_subscription_response(E3Interface._subscription_response_data(
        _event(cause=CAUSE_UNSUPPORTED_IDENTIFIER, detail="unknown control identifier 1")))
    assert iface.wait_for_subscription_result(2, timeout=0.05) is False
    assert iface.subscription_refusal(2) == {"cause": CAUSE_UNSUPPORTED_IDENTIFIER,
                                              "detail": "unknown control identifier 1"}
    iface.outbound_queue = queue.Queue()
    iface.send_subscription_request(7, 2, [1], [])
    assert iface.subscription_refusal(2) is None


def test_a_positive_answer_leaves_no_refusal():
    iface = _iface()
    iface._sub_reqid_to_rf[42] = 2
    iface._handle_subscription_response(E3Interface._subscription_response_data(_event(response_code=0)))
    assert iface.subscription_refusal(2) is None


# --- a disabled control is never sent ------------------------------------------

def test_a_disabled_control_is_not_queued():
    iface = _iface()
    iface.disable_control(2, 1)
    assert iface.control_disabled(2, 1)
    assert iface.schedule_control(7, 2, 1, b"x") is False
    assert iface.outbound_queue.empty()


def test_other_controls_are_still_queued():
    iface = _iface()
    iface.disable_control(2, 1)
    assert iface.schedule_control(7, 2, 2, b"x") is True      # same function, other control
    assert iface.schedule_control(7, 1, 1, b"x") is True      # same control id, other function
    assert iface.outbound_queue.qsize() == 2


# --- the dApp's decision ---------------------------------------------------------

class _FakeIface:
    def __init__(self, refusal, verdicts=(True,)):
        self.refusal = refusal
        self.verdicts = list(verdicts)
        self.disabled = []
        self.subscriptions = []

    def subscription_refusal(self, rf):
        return self.refusal

    def disable_control(self, rf, cid):
        self.disabled.append((rf, cid))

    def send_subscription_request(self, dapp_id, rf, telemetry, controls, sub_time=None, periodicity=None):
        self.subscriptions.append((rf, list(telemetry), list(controls)))
        return True

    def wait_for_subscription_result(self, rf, timeout=5.0):
        return self.verdicts.pop(0)

    # what the base class registers before it subscribes
    def add_subscription_callback(self, *a): pass
    def add_indication_callback(self, *a): pass
    def add_xapp_control_callback(self, *a): pass


def _dapp(fake, advertised):
    d = object.__new__(SpectrumSharingDApp)
    d.dapp_id = 7
    d.e3_interface = fake
    d.ran_function_controls = advertised
    return d


@pytest.mark.parametrize("cause", [CAUSE_UNSUPPORTED_IDENTIFIER, None])
def test_a_control_the_ran_does_not_offer_is_disabled_and_subscribed_again_without_it(cause):
    fake = _FakeIface({"cause": cause, "detail": "unknown control identifier 1"})
    d = _dapp(fake, {2: set()})
    assert d.retry_without_refused_controls(2, [1], [1]) is True
    assert fake.disabled == [(2, 1)]
    assert fake.subscriptions == [(2, [1], [])]


def test_only_the_controls_the_ran_lacks_are_dropped():
    fake = _FakeIface({"cause": CAUSE_UNSUPPORTED_IDENTIFIER, "detail": None})
    d = _dapp(fake, {1: {1}})
    assert d.retry_without_refused_controls(1, [1], [1, 2]) is True
    assert fake.disabled == [(1, 2)]
    assert fake.subscriptions == [(1, [1], [1])]


def test_another_cause_is_not_retried():
    fake = _FakeIface({"cause": 5, "detail": "RAN function 2 is not served"})   # unsupportedRanFunction
    d = _dapp(fake, {2: set()})
    assert d.retry_without_refused_controls(2, [1], [1]) is False
    assert fake.disabled == [] and fake.subscriptions == []


def test_a_refused_telemetry_id_is_not_retried():
    fake = _FakeIface({"cause": CAUSE_UNSUPPORTED_IDENTIFIER, "detail": "unknown telemetry identifier 9"})
    d = _dapp(fake, {2: {1}})                     # every control asked for is offered
    assert d.retry_without_refused_controls(2, [9], [1]) is False
    assert fake.disabled == [] and fake.subscriptions == []


def test_nothing_left_to_subscribe_to_is_not_retried():
    fake = _FakeIface({"cause": CAUSE_UNSUPPORTED_IDENTIFIER, "detail": None})
    d = _dapp(fake, {2: set()})
    assert d.retry_without_refused_controls(2, [], [1]) is False
    assert fake.subscriptions == []               # libe3 refuses a request with both lists empty


def test_without_the_setup_lists_nothing_is_guessed():
    fake = _FakeIface({"cause": CAUSE_UNSUPPORTED_IDENTIFIER, "detail": None})
    d = _dapp(fake, {})
    assert d.retry_without_refused_controls(2, [1], [1]) is False
    assert fake.disabled == []


def test_the_spectrum_dapp_keeps_going_when_the_l1_kpm_control_is_refused(monkeypatch):
    # RF=2 (L1-KPM) refuses control 1 and then accepts the telemetry alone; RF=1 is accepted.
    from dapp.dapp import DApp
    monkeypatch.setattr(DApp, "send_subscription_request", lambda self, *a, **k: True)  # not under test
    fake = _FakeIface({"cause": CAUSE_UNSUPPORTED_IDENTIFIER, "detail": "unknown control identifier 1"},
                      verdicts=[False, True, True])
    d = _dapp(fake, {2: set(), 1: {1, 2}})
    d.control = False
    assert d.send_subscription_request() is True
    assert fake.disabled == [(2, 1)]
    assert (2, [1], []) in fake.subscriptions


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
