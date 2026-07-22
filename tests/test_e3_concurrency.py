#!/usr/bin/env python3
"""Concurrency regression tests for E3Interface callback dispatch (issue #56)."""
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from e3interface.e3_interface import E3Interface


def _bare_interface():
    """An E3Interface with only the callback state, skipping connector setup."""
    iface = object.__new__(E3Interface)
    iface.indication_callbacks = {}
    iface.subscription_callbacks = {}
    iface.xapp_control_callbacks = {}
    iface._callback_lock = threading.Lock()
    iface.stop_event = threading.Event()
    iface.stop_event.set()  # keep __del__ from touching connections on GC
    return iface


def test_concurrent_add_remove_during_dispatch():
    """Churning the callback dict from one thread while another dispatches must
    not raise 'dictionary changed size during iteration'."""
    iface = _bare_interface()
    iface.add_indication_callback(1, 999, lambda d, rf, x: None)  # steady entry

    stop = threading.Event()
    errors = []

    def dispatcher():
        try:
            while not stop.is_set():
                iface._handle_indication_data(1, 2, b"payload")
        except Exception as e:  # noqa: BLE001 - record any race for the assert
            errors.append(e)

    t = threading.Thread(target=dispatcher)
    t.start()
    try:
        for i in range(3000):
            iface.add_indication_callback(1, i, lambda d, rf, x: None)
            iface.remove_indication_callback(1, i)
    finally:
        stop.set()
        t.join(timeout=5)

    assert not errors, f"dispatch raced with mutation: {errors!r}"


def test_dispatch_invokes_registered_callbacks():
    """Basic functional check that dispatch reaches the matching callbacks only."""
    iface = _bare_interface()
    hits = []
    iface.add_indication_callback(1, 1, lambda d, rf, x: hits.append((d, rf, x)))
    iface.add_indication_callback(2, 1, lambda d, rf, x: hits.append(("other", rf, x)))

    iface._handle_indication_data(1, 2, b"abc")

    assert hits == [(1, 2, b"abc")]


if __name__ == "__main__":
    test_concurrent_add_remove_during_dispatch()
    test_dispatch_invokes_registered_callbacks()
    print("e3 concurrency tests passed")
