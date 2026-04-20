#!/usr/bin/env python3
"""
Tests for Dashboard — label callback, initial_label, and emit_label.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from visualization.dashboard import Dashboard

# port=0 lets the OS assign a free ephemeral port — avoids collisions between tests
_PORT = 0


def test_label_callback_fires():
    received = []
    d = Dashboard(label_callback=received.append, initial_label="no_rfi", port=_PORT)
    assert d.label_callback is not None
    assert d.current_label == "no_rfi"
    d.label_callback("jammer")
    assert received == ["jammer"]
    d.stop()


def test_emit_label_updates_current_label():
    d = Dashboard(label_callback=lambda _: None, initial_label="no_rfi", port=_PORT)
    d.emit_label("radar")
    assert d.current_label == "radar"
    d.stop()


def test_no_label_callback_hides_selector():
    d = Dashboard(port=_PORT)
    assert d.label_callback is None
    assert d.current_label == ""
    d.stop()


def test_initial_label_empty_by_default():
    d = Dashboard(label_callback=lambda _: None, port=_PORT)
    assert d.current_label == ""
    d.stop()
