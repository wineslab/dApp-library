"""Operator-started SigMF recording of whole slots (all antennas, all symbols).

Independent of ``--save-iqs``, which keeps writing the detection-annotated
antenna-0 stream. Recording is started and stopped at runtime (dashboard), so
the writer is created on ``start`` and closed on ``stop``/expiry.

The inbound thread only calls :meth:`offer`, which is a flag check when idle
and a bounded ``put_nowait`` while recording; the IQSaver (disk I/O, close)
lives on a dedicated writer thread, so neither the inbound thread, the
detector worker nor the control thread ever waits on the disk.

On-disk layout: cf32_le, one record per slot, ``[ant][sym][sc][2]`` flattened,
``sc`` being the gNB's full 273-PRB slot (trailing PRBs beyond the cell
bandwidth are zero).
"""

from __future__ import annotations

import os
import queue
import shutil
import threading
import time
from typing import Any, Callable

import numpy as np

from e3interface.e3_logging import dapp_logger

QUEUE_DEPTH = 64
MIN_FREE_BYTES = 2 * 1024 ** 3
_FINALIZE_WAIT_S = 1.5
_DISK_CHECK_EVERY = 128


def _default_saver_factory(**kwargs):
    from iq_saver.iq_saver import IQSaver
    return IQSaver(**kwargs)


class SlotRecorder:
    def __init__(self, base_path: str, center_freq: float, fp16_beta: float,
                 subcarrier_spacing_khz: int = 30,
                 saver_factory: Callable[..., Any] | None = None) -> None:
        self._base_path = base_path
        self._center_freq = float(center_freq)
        self._fp16_beta = float(fp16_beta)
        self._scs_khz = int(subcarrier_spacing_khz)
        self._factory = saver_factory or _default_saver_factory
        self._lock = threading.Lock()
        self._accepting = False
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)
        self._thread: threading.Thread | None = None
        self._saver = None
        self._deadline = 0.0
        self._t0 = 0.0
        self._duration_s = 0.0
        self._filename = ""
        self._path = ""
        self._slots = 0
        self._dropped = 0
        self._bytes = 0
        self._sample_rate = 0.0
        self._stop_reason = ""
        self._finished = threading.Event()
        self._finished.set()

    # ------------------------------------------------------------------
    # capability

    def available(self) -> tuple[bool, str]:
        if self._factory is not _default_saver_factory:
            return True, ""
        try:
            import iq_saver.iq_saver  # noqa: F401
        except Exception as exc:
            return False, f"IQ writer not available ({exc.__class__.__name__}); install the dApp with its native extension"
        return True, ""

    # ------------------------------------------------------------------
    # control (called from the control thread)

    @property
    def recording(self) -> bool:
        return self._accepting

    def start(self, duration_s: float, notes: str = "") -> dict:
        with self._lock:
            if self._accepting or not self._finished.is_set():
                raise ValueError("already recording" if self._accepting
                                 else "previous recording is still being finalized")
            ok, why = self.available()
            if not ok:
                raise ValueError(why)
            free = shutil.disk_usage(self._base_path).free
            if free < MIN_FREE_BYTES:
                raise ValueError(
                    f"only {free // 2**20} MiB free under {self._base_path}")
            stem = "iq_full_" + time.strftime("%Y%m%d_%H%M%S")
            extra = {
                "domain": "frequency",
                "n_sc_per_prb": 12,
                "n_prbs": 273,
                "n_symbols": 14,
                "layout": "[ant][sym][prb][sc][I,Q]",
                "subcarrier_spacing_khz": self._scs_khz,
                "fp16_beta": self._fp16_beta,
                "all_antennas": True,
            }
            if notes:
                extra["user_notes"] = notes
            saver = self._factory(
                base_path=self._base_path,
                center_freq=self._center_freq,
                bandwidth=273 * 12 * self._scs_khz * 1e3,
                sample_rate=273 * 12 * self._scs_khz * 1e3,
                annotation_flush_interval=50,
                description="Full-slot capture from the SpectrumSharing dApp",
                dtype="cf32_le",
                filename=stem,
                **extra,
            )
            self._saver = saver
            self._filename = stem
            self._path = ""
            self._slots = 0
            self._dropped = 0
            self._bytes = 0
            self._stop_reason = ""
            self._duration_s = float(duration_s)
            self._t0 = time.monotonic()
            self._deadline = self._t0 + self._duration_s
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._finished.clear()
            self._thread = threading.Thread(
                target=self._writer, args=(saver,), name="dapp_rec", daemon=True)
            self._thread.start()
            self._accepting = True
        dapp_logger.info("Recording started: %s (%.1f s)", stem, duration_s)
        return {"ok": True, "filename": stem, "path": self._base_path,
                "duration_s": self._duration_s}

    def stop(self, reason: str = "stopped by user") -> dict:
        with self._lock:
            if not self._accepting:
                raise ValueError("not recording")
            self._accepting = False
            self._stop_reason = reason
            self._queue.put(None)
        self._finished.wait(timeout=_FINALIZE_WAIT_S)
        done = self._finished.is_set()
        return {"ok": True, "reason": reason, "filename": self._filename,
                "slots": self._slots, "bytes": self._bytes,
                "dropped": self._dropped, "finalized": done}

    def status(self) -> dict:
        recording = self._accepting
        elapsed = (time.monotonic() - self._t0) if recording else 0.0
        return {
            "state": "recording" if recording else "idle",
            "filename": self._filename,
            "bytes": self._bytes,
            "slots": self._slots,
            "dropped": self._dropped,
            "elapsed_s": elapsed,
            "duration_s": self._duration_s if recording else 0.0,
        }

    def tick(self) -> None:
        """Duration expiry; called from the control thread's poll loop."""
        if self._accepting and time.monotonic() >= self._deadline:
            try:
                self.stop("duration limit reached")
            except ValueError:
                pass

    def close(self) -> None:
        if self._accepting:
            try:
                self.stop("dApp shutdown")
            except ValueError:
                pass
        self._finished.wait(timeout=5.0)

    # ------------------------------------------------------------------
    # data path

    def offer(self, full: np.ndarray, timestamp: float, sfn: int, slot: int,
              ranges, valid_symbol_mask: int) -> None:
        if not self._accepting:
            return
        try:
            self._queue.put_nowait(
                (full, timestamp, sfn, slot, ranges, valid_symbol_mask))
        except queue.Full:
            self._dropped += 1

    def _writer(self, saver) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                full, ts, sfn, slot, ranges, vmask = item
                try:
                    flat = np.ascontiguousarray(full, dtype=np.float32) \
                        .view(np.complex64).reshape(-1)
                    start = saver.save_samples(flat)
                    saver.add_annotation(
                        start_sample=start,
                        label="slot",
                        timestamp_ns=int(ts),
                        sfn=int(sfn),
                        slot=int(slot),
                        n_ants=int(full.shape[0]),
                        valid_symbol_mask=int(vmask),
                        sensing_ranges=[
                            {"start_symbol": int(r.start_symbol),
                             "num_symbols": int(r.num_symbols),
                             "rb_start": int(r.rb_start),
                             "rb_size": int(r.rb_size)}
                            for r in (ranges or ())
                        ],
                    )
                    self._slots += 1
                    self._bytes += flat.size * 8
                    if (self._slots % _DISK_CHECK_EVERY == 0
                            and shutil.disk_usage(self._base_path).free < MIN_FREE_BYTES):
                        dapp_logger.warning("recording stopped: low disk space")
                        self._accepting = False
                        self._stop_reason = "low disk space"
                        break
                except Exception:
                    dapp_logger.exception("recording write failed; stopping")
                    self._accepting = False
                    self._stop_reason = "write error"
                    break
        finally:
            try:
                info = saver.get_recording_info()
                paths = info.get("file_paths") or []
                if paths:
                    self._path = paths[-1]
                    self._filename = os.path.basename(self._path)
                sizes = info.get("file_size_bytes") or {}
                if sizes:
                    self._bytes = int(sum(sizes.values()))
            except Exception:
                pass
            try:
                saver.close()
            except Exception:
                dapp_logger.exception("closing the recording failed")
            dapp_logger.info(
                "Recording finished (%s): %s, %d slots, %d bytes, %d dropped",
                self._stop_reason or "done", self._path or self._filename,
                self._slots, self._bytes, self._dropped)
            self._saver = None
            self._finished.set()
