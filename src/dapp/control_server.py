"""ZMQ REP server for the dashboard's control channel.

The subcarrier dashboard proxies every control click as one JSON request
(``{"cmd": ...}``) over a REQ socket and waits at most 2 s for one JSON reply.
This module only moves JSON: the semantics live in the handlers a dApp
registers, so it knows nothing about spectrum sensing.

Binds loopback only. The socket is bound in the constructor, so a port that is
already in use is reported to the caller before anything else is started.
"""

from __future__ import annotations

import json
import threading
from typing import Callable

from e3interface.e3_logging import dapp_logger

Handler = Callable[[dict], dict]

_POLL_MS = 200


class ControlBindError(RuntimeError):
    """The control port could not be bound (typically: already in use)."""


class ControlServer:
    """REP server; one daemon thread, one reply per request, always."""

    def __init__(self, port: int, host: str = "127.0.0.1",
                 tick: Callable[[], None] | None = None) -> None:
        import zmq  # lazy: pyzmq is an optional extra

        self._zmq = zmq
        self._handlers: dict[str, Handler] = {}
        self._tick = tick
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.setsockopt(zmq.LINGER, 0)
        try:
            if int(port) == 0:
                # let the OS pick: there is no window between choosing a free port and binding it
                port = self._sock.bind_to_random_port(f"tcp://{host}")
            else:
                self._sock.bind(f"tcp://{host}:{int(port)}")
        except zmq.ZMQError as exc:
            self._sock.close(0)
            self._ctx.term()
            raise ControlBindError(
                f"cannot bind control port tcp://{host}:{port}: {exc}") from exc
        self.host = host
        self.port = int(port)

    def register(self, cmd: str, handler: Handler) -> None:
        self._handlers[cmd] = handler

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="dapp_ctrl", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self._sock.close(0)
            self._ctx.term()
        except Exception:
            pass

    def dispatch(self, raw: str) -> dict:
        """Turn one request string into a reply dict (never raises)."""
        try:
            req = json.loads(raw)
        except ValueError as exc:
            return {"ok": False, "error": f"parse: {exc}"}
        if not isinstance(req, dict):
            return {"ok": False, "error": "parse: request must be a JSON object"}
        cmd = req.get("cmd")
        handler = self._handlers.get(cmd) if isinstance(cmd, str) else None
        if handler is None:
            return {"ok": False, "error": f"unknown cmd: {cmd}"}
        try:
            reply = handler(req)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:
            dapp_logger.exception("control handler %r failed", cmd)
            return {"ok": False, "error": f"internal: {exc}"}
        return reply

    def _loop(self) -> None:
        zmq = self._zmq
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        while not self._stop.is_set():
            if self._tick is not None:
                try:
                    self._tick()
                except Exception:
                    dapp_logger.exception("control tick failed")
            if not dict(poller.poll(_POLL_MS)):
                continue
            try:
                raw = self._sock.recv_string()
            except zmq.ZMQError:
                continue
            reply = self.dispatch(raw)
            try:
                self._sock.send_string(json.dumps(reply))
            except (zmq.ZMQError, TypeError, ValueError):
                # An unserialisable reply must still be answered, or the REP
                # socket stays in the "must send" state and wedges.
                try:
                    self._sock.send_string(json.dumps(
                        {"ok": False, "error": "internal: bad reply"}))
                except zmq.ZMQError:
                    pass
