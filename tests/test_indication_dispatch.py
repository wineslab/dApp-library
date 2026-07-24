#!/usr/bin/env python3
"""Dispatch-by-ranFunctionId and PRB-block reconciliation regression tests.

These cover the two behaviour changes in the DAppControlData envelope release:

  * indications are routed by ranFunctionId (RF=1 Spectrum sensing vs RF=2
    L1-KPM IQ), not by trial-parsing the payload — a Spectrum indication that
    happens to decode as a valid L1-KPM one must NOT reach the IQ path;
  * PRB blocks use the gNB's REPLACE semantics: every change re-sends the full
    reconciled union of the detection and xApp sources, and neither source
    clobbers the other.
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import spectrum.spectrum_dapp as sd
from spectrum.spectrum_dapp import SpectrumSharingDApp
from e3interface.e3_interface import E3Interface


def _bare_dapp():
    """A SpectrumSharingDApp with only the fields the tested methods touch."""
    d = object.__new__(SpectrumSharingDApp)
    d.dapp_id = 7
    return d


class _FakeIface:
    def __init__(self):
        self.controls = []

    def schedule_control(self, *, dappId, ranFunctionId, controlId, actionData):
        self.controls.append((ranFunctionId, controlId, actionData))


def test_dispatch_routes_by_ran_function_id(monkeypatch):
    d = _bare_dapp()
    calls = {"l1": [], "sensing": []}
    d._handle_l1_indication = lambda p, t: calls["l1"].append(p)
    d._handle_sensing_indication = lambda data: (calls["sensing"].append(data) or True)

    # Force the L1 pointer parse to always "succeed" — this is the ~25% false
    # decode that used to mis-route RF=1 payloads into the IQ path.
    class _AlwaysPointer:
        @staticmethod
        def from_bytes(data):
            return object()
    monkeypatch.setattr(sd, "SlotPointer", _AlwaysPointer)

    # RF=1 (Spectrum sensing) must go to the sensing handler, never L1 — even
    # though from_bytes would return a (bogus) pointer.
    d._handle_indication(d.dapp_id, SpectrumSharingDApp.PRB_CONTROL_RAN_FUNCTION_ID, b"x")
    assert len(calls["sensing"]) == 1
    assert calls["l1"] == []

    # RF=2 (L1-KPM) goes to the IQ path.
    d._handle_indication(d.dapp_id, SpectrumSharingDApp.RAN_FUNCTION_ID, b"y")
    assert len(calls["l1"]) == 1
    assert len(calls["sensing"]) == 1  # unchanged

    # An unexpected RF id is dropped.
    d._handle_indication(d.dapp_id, 99, b"z")
    assert len(calls["l1"]) == 1 and len(calls["sensing"]) == 1


def _sent_sets(iface):
    """Decode the blocked set from each captured control (create_prb_block_control
    is patched to return the sorted list bytes)."""
    return [set(eval(a.decode())) if a else set() for (_, _, a) in iface.controls]


def test_prb_reconcile_replace_semantics():
    d = _bare_dapp()
    d._prb_block_lock = threading.Lock()
    d._prb_block_detect = set()
    d._prb_block_xapp = set()
    d._prb_block_sent = set()
    d.e3_interface = _FakeIface()
    # Encode the full blocked list into the actionData so the test can read it.
    d.create_prb_block_control = lambda blocked_prbs, update_sampling=False: repr(
        sorted(blocked_prbs)
    ).encode()

    # Detection blocks {1,2}: full set sent.
    d._reconcile_prb_blocks(detect={1, 2})
    assert d._prb_block_sent == {1, 2}

    # xApp adds {5}: the union is re-sent, detection's PRBs preserved.
    d._reconcile_prb_blocks(xapp={5})
    assert d._prb_block_sent == {1, 2, 5}

    # Detection clears: xApp's {5} must NOT be unblocked (REPLACE would wipe it
    # if we sent a delta).
    d._reconcile_prb_blocks(detect=set())
    assert d._prb_block_sent == {5}

    # No change → no extra control emitted.
    n_before = len(d.e3_interface.controls)
    d._reconcile_prb_blocks(detect=set())
    assert len(d.e3_interface.controls) == n_before

    # Every emitted control is the FULL set (never a bare delta).
    assert _sent_sets(d.e3_interface) == [{1, 2}, {1, 2, 5}, {5}]


def test_clear_prb_blocks_is_unconditional():
    d = _bare_dapp()
    d._prb_block_lock = threading.Lock()
    d._prb_block_detect = set()
    d._prb_block_xapp = set()
    d._prb_block_sent = set()  # already empty — a stale gNB may still hold blocks
    d.e3_interface = _FakeIface()
    d.create_prb_block_control = lambda blocked_prbs, update_sampling=False: repr(
        sorted(blocked_prbs)
    ).encode()

    assert d.clear_prb_blocks() is True
    # One control sent even though _prb_block_sent was empty, and it is a clear.
    assert len(d.e3_interface.controls) == 1
    assert _sent_sets(d.e3_interface) == [set()]


def _bare_iface():
    """An E3Interface with only the subscription-correlation state."""
    iface = object.__new__(E3Interface)
    iface.subscription_callbacks = {}
    iface._callback_lock = threading.Lock()
    iface._sub_cv = threading.Condition()
    iface._sub_pending = {}
    iface._sub_results = {}
    iface.stop_event = threading.Event()
    iface.stop_event.set()  # keep __del__ from touching connections on GC
    return iface


def test_subscription_result_tied_to_gnb_response():
    """A queued request is not an accepted one: wait_for_subscription_result
    reflects the gNB's SubscriptionResponse (by requestId), or None on timeout."""
    iface = _bare_iface()

    # No response yet for a pending RF=1 request → timeout returns None.
    with iface._sub_cv:
        iface._sub_pending[42] = 1
    assert iface.wait_for_subscription_result(1, timeout=0.05) is None

    # Positive response (requestId 42) → accepted.
    iface._handle_subscription_response(
        {"dAppIdentifier": 7, "requestId": 42, "responseCode": "positive", "subscriptionId": 3}
    )
    assert iface.wait_for_subscription_result(1, timeout=0.05) is True

    # Negative response for a different RF → rejected.
    with iface._sub_cv:
        iface._sub_pending[43] = 2
    iface._handle_subscription_response(
        {"dAppIdentifier": 7, "requestId": 43, "responseCode": "negative"}
    )
    assert iface.wait_for_subscription_result(2, timeout=0.05) is False


def test_subscription_wait_unblocks_on_async_response():
    """A response arriving on another thread unblocks a waiter promptly."""
    iface = _bare_iface()
    with iface._sub_cv:
        iface._sub_pending[7] = 1

    def responder():
        time.sleep(0.05)
        iface._handle_subscription_response(
            {"dAppIdentifier": 1, "requestId": 7, "responseCode": "positive"}
        )

    t = threading.Thread(target=responder)
    t.start()
    try:
        assert iface.wait_for_subscription_result(1, timeout=2.0) is True
    finally:
        t.join(timeout=2)
