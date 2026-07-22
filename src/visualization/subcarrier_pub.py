"""Publish per-slot subcarrier power to the vendored subcarrier visualizer.

The visualizer (``visualization/subcarrier/subcarrier_visualizer.py``) is a
standalone Flask + WebSocket app whose ZMQ SUB connects to this PUB. Wire
format (multipart), matching the visualizer's zmq_receiver:

    frame 0: "subcarrier_power|sfn|slot|n_ant|n_sym|n_sc|u8|db_min|db_max|
              det=0|det_n=0|det_thr=0|kernel_us=0"
    frame 1: n_ant*n_sym*n_sc bytes of u8 power (quantized over [db_min, db_max])
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import zmq

from e3interface.e3_logging import dapp_logger

_VISUALIZER = os.path.join(os.path.dirname(__file__), "subcarrier", "subcarrier_visualizer.py")


class SubcarrierPublisher:
    """ZMQ PUB feeding the subcarrier visualizer; optionally spawns it."""

    def __init__(self, zmq_port: int = 5559, web_port: int = 5001,
                 db_min: int = 20, db_max: int = 90, spawn_viz: bool = True):
        self._db_min = int(db_min)
        self._db_max = int(db_max)
        self._span = max(1.0, float(self._db_max - self._db_min))
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.set_hwm(8)
        self._sock.bind(f"tcp://*:{zmq_port}")
        self._viz = self._spawn(web_port, zmq_port) if spawn_viz else None
        dapp_logger.info(
            "SubcarrierPublisher bound tcp://*:%d (visualizer %s, web :%d)",
            zmq_port, "spawned" if self._viz else "external", web_port,
        )

    def _spawn(self, web_port: int, zmq_port: int):
        try:
            return subprocess.Popen(
                [sys.executable, _VISUALIZER, "--port", str(web_port),
                 "--zmq-port", str(zmq_port)]
            )
        except Exception:
            dapp_logger.exception("Failed to spawn subcarrier visualizer")
            return None

    def publish_slot(self, mag_2d: np.ndarray, sfn: int, slot: int) -> None:
        """Publish one antenna's ``[n_sym][n_sc]`` magnitudes as a u8 dB grid."""
        m = np.asarray(mag_2d, dtype=np.float32)
        if m.ndim != 2:
            return
        n_sym, n_sc = m.shape
        db = 20.0 * np.log10(np.maximum(m, 1e-6))
        u8 = np.clip((db - self._db_min) * (255.0 / self._span), 0, 255).astype(np.uint8)
        meta = (
            f"subcarrier_power|{sfn}|{slot}|1|{n_sym}|{n_sc}|u8|"
            f"{self._db_min}|{self._db_max}|det=0|det_n=0|det_thr=0|kernel_us=0"
        ).encode()
        try:
            self._sock.send_multipart([meta, u8.tobytes()], flags=zmq.NOBLOCK)
        except zmq.ZMQError:
            pass

    def stop(self) -> None:
        try:
            self._sock.close(0)
        except Exception:
            pass
        if self._viz is not None:
            self._viz.terminate()
            self._viz = None
