"""Publish per-slot subcarrier power to the vendored subcarrier visualizer.

The visualizer (``visualization/subcarrier/subcarrier_visualizer.py``) is a
standalone Flask + WebSocket app whose ZMQ SUB connects to this PUB. Wire
format (multipart), matching the visualizer's zmq_receiver:

    frame 0: "subcarrier_power|sfn|slot|n_ant|n_sym|n_sc|u8|db_min|db_max|
              det=<0|1>|det_n=<n>|det_thr=<db>|kernel_us=0"
    frame 1: n_ant*n_sym*n_sc bytes of u8 power (quantized over [db_min, db_max])
    frame 2: (only when det=1) n_sc bytes of u8 detection mask (1 = blocked)
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import threading

import numpy as np
import zmq

from e3interface.e3_logging import dapp_logger

_VISUALIZER = os.path.join(os.path.dirname(__file__), "subcarrier", "subcarrier_visualizer.py")


def _child_preexec():
    """Run in the spawned child before exec (Linux only).

    Ask the kernel to SIGKILL this child when the parent dies (PR_SET_PDEATHSIG)
    so a hard SIGKILL of the dApp — which atexit/signal handlers can't catch —
    still can't orphan the visualizer holding web :5001. Best-effort: any failure
    is ignored (start_new_session + stop() still cover the graceful paths)."""
    try:
        import ctypes
        import ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    except Exception:
        pass


SC_PER_PRB = 12


class SubcarrierPublisher:
    """ZMQ PUB feeding the subcarrier visualizer; optionally spawns it.

    ``ctrl_port`` is the dApp's control REP the spawned dashboard proxies its
    control panels to (0 disables them).
    """

    def __init__(self, zmq_port: int = 5559, web_port: int = 5001,
                 num_prbs: int = 273, db_min: int = 20, db_max: int = 90,
                 spawn_viz: bool = True, ctrl_port: int = 0,
                 center_freq: float = 0.0, scs_khz: float = 30.0):
        self._ctrl_port = int(ctrl_port)
        self._center_freq = float(center_freq)
        self._scs_khz = float(scs_khz)
        # (freq_gran, time_gran, agg, subsample_n), swapped as one tuple.
        self._viz_cfg = ("sc", "sym", "mean", 1)
        self._viz_lock = threading.Lock()
        self._slot_counter = 0
        self._db_min = int(db_min)
        self._db_max = int(db_max)
        self._span = max(1.0, float(self._db_max - self._db_min))
        self._num_prbs = int(num_prbs)
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.set_hwm(8)
        try:
            self._sock.bind(f"tcp://*:{zmq_port}")
        except zmq.ZMQError as exc:
            # 5559 is also the C++ dApp's results-PUB port; a co-resident dApp,
            # a prior instance, or an orphan will hold it. Don't abort dApp init.
            self._sock.close(0)
            self._sock = None
            self._viz = None
            dapp_logger.error(
                "SubcarrierPublisher: could not bind tcp://*:%d (%s); "
                "the subcarrier visualizer will be disabled for this run",
                zmq_port, exc,
            )
            return
        self._viz = self._spawn(web_port, zmq_port) if spawn_viz else None
        if self._viz is not None:
            # Cover graceful teardown (normal exit, sys.exit, unhandled
            # exception); the child's PR_SET_PDEATHSIG covers a hard kill.
            atexit.register(self.stop)
        dapp_logger.info(
            "SubcarrierPublisher bound tcp://*:%d (visualizer %s, web :%d, %d PRB)",
            zmq_port, "spawned" if self._viz else "external", web_port, self._num_prbs,
        )

    def _spawn(self, web_port: int, zmq_port: int):
        if not os.path.isfile(_VISUALIZER):
            # In a wheel install visualization/subcarrier/ may not ship.
            dapp_logger.error("Subcarrier visualizer not found at %s; not spawning", _VISUALIZER)
            return None
        try:
            # The dashboard proxies its control panels to this dApp's own REP
            # on loopback; port 0 leaves them disabled. start_new_session puts the child in its own process group so stop()
            # can kill the whole group; preexec sets PR_SET_PDEATHSIG on Linux.
            proc = subprocess.Popen(
                [sys.executable, _VISUALIZER, "--port", str(web_port),
                 "--zmq-port", str(zmq_port), "--num-prbs", str(self._num_prbs),
                 "--center-freq", str(self._center_freq),
                 "--scs-khz", str(self._scs_khz),
                 "--ctrl-port", str(self._ctrl_port),
                 "--ctrl-host", "127.0.0.1"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
                preexec_fn=_child_preexec if sys.platform.startswith("linux") else None,
            )
        except Exception:
            dapp_logger.exception("Failed to spawn subcarrier visualizer")
            return None
        if proc.poll() is not None:
            dapp_logger.error("Subcarrier visualizer exited immediately (rc=%s)", proc.returncode)
            return None
        return proc

    def get_viz_config(self) -> dict:
        freq, time_, agg, sub = self._viz_cfg
        return {"freq_gran": freq, "time_gran": time_, "agg": agg,
                "subsample_n": sub}

    def set_viz_config(self, freq_gran: str | None = None,
                       time_gran: str | None = None, agg: str | None = None,
                       subsample_n: int | None = None) -> dict:
        """Change what is published; validates everything before applying."""
        with self._viz_lock:
            freq, time_, cur_agg, sub = self._viz_cfg
            if freq_gran is not None:
                if freq_gran not in ("sc", "prb"):
                    raise ValueError("freq_gran must be 'sc' or 'prb'")
                freq = freq_gran
            if time_gran is not None:
                if time_gran not in ("sym", "slot"):
                    raise ValueError("time_gran must be 'sym' or 'slot'")
                time_ = time_gran
            if agg is not None:
                if agg not in ("mean", "max"):
                    raise ValueError("agg must be 'mean' or 'max'")
                cur_agg = agg
            if subsample_n is not None:
                if int(subsample_n) < 1:
                    raise ValueError("subsample_n must be >= 1")
                sub = int(subsample_n)
            self._viz_cfg = (freq, time_, cur_agg, sub)
        return self.get_viz_config()

    @staticmethod
    def _reduce(a: np.ndarray, axis: int, agg: str) -> np.ndarray:
        return a.max(axis=axis) if agg == "max" else a.mean(axis=axis)

    def publish_slot(self, mag_2d: np.ndarray, sfn: int, slot: int,
                     blocked: np.ndarray | None = None,
                     det_thr: float | None = None) -> None:
        """Publish one antenna's ``[n_sym][n_sc]`` magnitudes as a u8 dB grid.

        ``blocked`` is an optional per-subcarrier detection mask (length n_sc);
        when given, the detection frame is appended and the visualizer draws the
        red block strip. ``det_thr`` is the detector threshold in dB (shown in
        the stats).
        """
        if self._sock is None:
            return
        freq_gran, time_gran, agg, subsample = self._viz_cfg
        self._slot_counter += 1
        if subsample > 1 and (self._slot_counter % subsample) != 1:
            return
        m = np.asarray(mag_2d, dtype=np.float32)
        if m.ndim != 2:
            return
        # A stale/mis-dispatched slot can carry non-finite IQ; log10 of that
        # poisons the whole frame. Replace before quantizing.
        if not np.isfinite(m).all():
            m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        mask_in = None if blocked is None else np.asarray(blocked).reshape(-1)
        if time_gran == "slot":
            m = self._reduce(m, 0, agg).reshape(1, -1)
        if freq_gran == "prb" and m.shape[1] % SC_PER_PRB == 0:
            m = self._reduce(m.reshape(m.shape[0], -1, SC_PER_PRB), 2, agg)
            if mask_in is not None and mask_in.size == m.shape[1] * SC_PER_PRB:
                mask_in = (mask_in != 0).reshape(-1, SC_PER_PRB).any(axis=1)
        n_sym, n_sc = m.shape
        db = 20.0 * np.log10(np.maximum(m, 1e-6))
        u8 = np.clip((db - self._db_min) * (255.0 / self._span), 0, 255).astype(np.uint8)

        det = 0
        det_n = 0
        thr = 0.0
        mask_bytes = b""
        if mask_in is not None:
            mask = mask_in
            if mask.size == n_sc:
                det = 1
                det_n = int(n_sc)
                thr = float(det_thr) if det_thr is not None else 0.0
                mask_bytes = (mask != 0).astype(np.uint8).tobytes()
        meta = (
            f"subcarrier_power|{sfn}|{slot}|1|{n_sym}|{n_sc}|u8|"
            f"{self._db_min}|{self._db_max}|det={det}|det_n={det_n}|"
            f"det_thr={thr:.1f}|kernel_us=0"
        ).encode()
        parts = [meta, u8.tobytes()]
        if det:
            parts.append(mask_bytes)
        try:
            self._sock.send_multipart(parts, flags=zmq.NOBLOCK)
        except zmq.ZMQError:
            pass

    def stop(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close(0)
            except Exception:
                pass
            self._sock = None
        if self._viz is not None:
            # Signal the child's whole process group (start_new_session made it a
            # group leader) so any grandchildren die too and nothing keeps :5001.
            try:
                pgid = os.getpgid(self._viz.pid)
            except (ProcessLookupError, OSError):
                pgid = None

            def _sig_group(sig):
                if pgid is not None:
                    try:
                        os.killpg(pgid, sig)
                    except (ProcessLookupError, OSError):
                        pass
                else:
                    try:
                        self._viz.send_signal(sig)
                    except (ProcessLookupError, OSError):
                        pass

            _sig_group(signal.SIGTERM)
            try:
                self._viz.wait(timeout=3)
            except subprocess.TimeoutExpired:
                _sig_group(signal.SIGKILL)
                try:
                    self._viz.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    dapp_logger.warning("Subcarrier visualizer did not exit after kill()")
            self._viz = None
