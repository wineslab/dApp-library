#!/usr/bin/env python3
"""ControlServer: one reply per request, loopback bind, bind failure reported."""
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

zmq = pytest.importorskip("zmq")

from dapp.control_server import ControlBindError, ControlServer


@pytest.fixture
def server():
    srv = ControlServer(0)
    srv.register("echo", lambda req: {"ok": True, "got": req.get("v")})
    srv.register("bad", lambda req: (_ for _ in ()).throw(ValueError("nope")))
    srv.register("boom", lambda req: 1 / 0)
    srv.register("unserialisable", lambda req: {"ok": True, "x": object()})
    srv.start()
    yield srv
    srv.stop()


def _req(port, payload, raw=False):
    sock = zmq.Context.instance().socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    sock.connect(f"tcp://127.0.0.1:{port}")
    try:
        sock.send_string(payload if raw else json.dumps(payload))
        return json.loads(sock.recv_string())
    finally:
        sock.close(0)


def test_dispatch_and_errors(server):
    assert _req(server.port, {"cmd": "echo", "v": 3}) == {"ok": True, "got": 3}
    assert _req(server.port, {"cmd": "nope"}) == {"ok": False, "error": "unknown cmd: nope"}
    assert _req(server.port, {"cmd": "bad"}) == {"ok": False, "error": "nope"}
    r = _req(server.port, {"cmd": "boom"})
    assert r["ok"] is False and r["error"].startswith("internal:")


def test_malformed_requests_still_get_one_reply(server):
    assert _req(server.port, "{not json", raw=True)["error"].startswith("parse:")
    assert _req(server.port, "[1, 2]", raw=True)["error"].startswith("parse:")
    # The socket is not wedged afterwards.
    assert _req(server.port, {"cmd": "echo", "v": 1})["ok"] is True


def test_unserialisable_reply_does_not_wedge(server):
    assert _req(server.port, {"cmd": "unserialisable"})["error"].startswith("internal:")
    assert _req(server.port, {"cmd": "echo", "v": 2})["got"] == 2


def test_bind_in_use_is_reported():
    held = zmq.Context.instance().socket(zmq.REP)
    port = held.bind_to_random_port("tcp://127.0.0.1")
    try:
        with pytest.raises(ControlBindError):
            ControlServer(port)
    finally:
        held.close(0)


def test_binds_loopback_only():
    srv = ControlServer(0)
    try:
        assert srv.host == "127.0.0.1"
    finally:
        srv.stop()


def test_tick_runs_and_stop_joins():
    ticks = []
    srv = ControlServer(0, tick=lambda: ticks.append(1))
    srv.start()
    time.sleep(0.5)
    srv.stop()
    assert len(ticks) >= 2
    assert srv._thread is None


def test_port_zero_binds_a_random_port_and_reports_it():
    a, b = ControlServer(0), ControlServer(0)
    try:
        assert a.port > 0 and b.port > 0 and a.port != b.port
    finally:
        a.stop()
        b.stop()
