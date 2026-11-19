#!/usr/bin/env python3
"""The periodicity of a subscription response reaches the dApp as the RAN reported it.

libe3 0.4.3 answers with the least time, in microseconds, the RAN keeps between two
indications, or says nothing. libe3py shows nothing as -1; the dApp sees None.
"""
import logging
import os
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from e3interface.e3_interface import E3Interface
from spectrum.spectrum_dapp import SpectrumSharingDApp


def _event(periodicity, response_code=0):
    return SimpleNamespace(
        dapp_id=7, response_code=response_code, subscription_id=3,
        request_id=42, periodicity=periodicity,
    )


@pytest.mark.parametrize("reported, expected", [(-1, None), (1000, 1000), (0, 0)])
def test_periodicity_is_none_only_when_the_ran_did_not_say(reported, expected):
    data = E3Interface._subscription_response_data(_event(reported))
    assert data["periodicity"] == expected


def test_response_data_keeps_what_it_carried_before():
    data = E3Interface._subscription_response_data(_event(1000))
    assert data["dAppIdentifier"] == 7
    assert data["responseCode"] == "positive"
    assert data["subscriptionId"] == 3
    assert data["requestId"] == 42
    assert E3Interface._subscription_response_data(_event(-1, 1))["responseCode"] == "negative"


def test_callback_receives_the_periodicity():
    iface = object.__new__(E3Interface)
    iface.subscription_callbacks = {}
    iface._callback_lock = threading.Lock()
    iface._sub_cv = threading.Condition()
    iface._sub_reqid_to_rf = {}
    iface._sub_pending_rfs = set()
    iface._sub_results = {}
    iface.stop_event = threading.Event()
    iface.stop_event.set()  # keep __del__ from touching connections on GC
    got = []
    iface.subscription_callbacks[7] = [got.append]

    iface._handle_subscription_response(E3Interface._subscription_response_data(_event(1000)))
    iface._handle_subscription_response(E3Interface._subscription_response_data(_event(-1)))

    assert [d["periodicity"] for d in got] == [1000, None]


class _FakeIface:
    def add_indication_callback(self, *a):
        pass

    def add_xapp_control_callback(self, *a):
        pass


def _dapp():
    d = object.__new__(SpectrumSharingDApp)
    d.dapp_id = 7
    d.e3_interface = _FakeIface()
    d._handle_indication = lambda *a: None
    d._handle_xapp_control = lambda *a: None
    return d


def _response(periodicity, code="positive"):
    return {"dAppIdentifier": 7, "responseCode": code, "subscriptionId": 3,
            "requestId": 42, "periodicity": periodicity}


def test_dapp_logs_a_missing_periodicity(caplog):
    with caplog.at_level(logging.INFO, logger="dapp_logger"):
        _dapp().manage_subscription_response(_response(None))
    assert "periodicity not reported by the RAN" in caplog.text


def test_dapp_logs_the_periodicity_the_ran_keeps(caplog):
    with caplog.at_level(logging.INFO, logger="dapp_logger"):
        _dapp().manage_subscription_response(_response(1000))
    assert "keeps at least 1000 us between indications" in caplog.text
    assert "not reported" not in caplog.text


def test_dapp_still_registers_callbacks_without_a_periodicity():
    d = _dapp()
    seen = []
    d.e3_interface.add_indication_callback = lambda *a: seen.append(a)
    d.manage_subscription_response(_response(None))
    assert seen and seen[0][1] == 3


def test_a_negative_response_says_nothing_of_the_interval(caplog):
    with caplog.at_level(logging.INFO, logger="dapp_logger"):
        _dapp().manage_subscription_response(_response(None, "negative"))
    assert "not reported" not in caplog.text
    assert "keeps at least" not in caplog.text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
