#!/usr/bin/env python3
"""Smoke tests for the libe3-backed E3AP layer.

Skipped automatically if libe3py is not installed in this environment
(build it with `./build_libe3 --install --enable-swig`).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from e3interface import libe3_agent  # noqa: E402  (module imports without libe3py)

Libe3Agent = libe3_agent.Libe3Agent

pytestmark = pytest.mark.skipif(
    not libe3_agent.LIBE3PY_AVAILABLE,
    reason="libe3py not installed; build with build_libe3 --install --enable-swig",
)


def test_agent_constructs_and_polls_empty():
    """Constructing an agent (no peer) works; poll_events blocks then returns []."""
    import time

    agent = Libe3Agent(link="zmq", transport="ipc", encoding="asn1",
                       dapp_name="TestDApp", log_level=0)
    assert agent.dapp_id is None
    assert agent.dropped_events() == 0

    t0 = time.monotonic()
    batch = agent.poll_events(64, 120)
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    assert len(list(batch)) == 0
    assert elapsed_ms >= 90  # blocked for ~the timeout, GIL released


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        Libe3Agent(link="carrier-pigeon", transport="ipc", encoding="asn1")
    with pytest.raises(ValueError):
        Libe3Agent(link="zmq", transport="ipc", encoding="morse")


def test_event_kind_constants_exposed():
    for name in ("EVENT_INDICATION", "EVENT_XAPP_CONTROL",
                 "EVENT_SUBSCRIPTION_RESPONSE", "EVENT_SETUP_RESPONSE",
                 "EVENT_MESSAGE_ACK"):
        assert hasattr(libe3_agent, name)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
