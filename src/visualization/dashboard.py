"""Web dashboard for the spectrum-sensing dApp.

Threading model:
  - Flask/SocketIO server thread (serves the static page + websocket).
  - Emit-loop thread, ~60 Hz, drains _pending_mag_rows → wire under a
    visibility gate and K-in-flight ack flow control.
  - Producer (process_iq_data, called by the dApp's indication worker)
    fills _pending_mag_rows; coalesce-on-overflow keeps it bounded.

Wire format for magnitude rows is uint16-quantized dB over a fixed
[QUANT_MIN_DB, QUANT_MAX_DB] range, matched by the browser worker.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass

try:
    from flask import Flask, render_template
    from flask_socketio import SocketIO
except ModuleNotFoundError:
    print(
        "Optional dependencies for GUI not installed.\n"
        "    pip install 'dApps[gui]'"
    )
    exit(-1)
import numpy as np

from e3interface.e3_logging import dapp_logger


# Wire-format quantization: dB values clipped to [QUANT_MIN_DB, QUANT_MAX_DB]
# and packed as uint16. 2 bytes/bin vs 4 for Float32, 1.83 mdB/LSB resolution
# — invisible on a 256-stop palette. The browser worker dequantizes inline
# in its palette-mapping multiply.
QUANT_MIN_DB = -20.0
QUANT_MAX_DB = 100.0
QUANT_SCALE = 65535.0 / (QUANT_MAX_DB - QUANT_MIN_DB)

# Emit-loop tick frequency. The actual emit rate (call_hz in logs) is gated
# by browser acks; this is just the max decision-check rate. 60 Hz matches
# typical browser rAF and gives natural batching of producer rows.
DEFAULT_EMIT_TARGET_FPS = 60.0
STATS_INTERVAL_S = 5.0
MAX_INFLIGHT = 3                       # K-in-flight ack: ~50 ms RTT jitter budget
EMIT_WATCHDOG_NS = 500_000_000         # retire unacked emits after 500 ms
MAX_PENDING_MAG_ROWS = 128             # coalesce kicks in at this many queued rows


@dataclass
class _RowCounters:
    """Cumulative row-accounting since startup (surfaced in stats log)."""
    submitted: int = 0       # entered _submit_magnitude_blob
    emitted: int = 0         # actually shipped over socket.io
    coalesced: int = 0       # mean-folded by _trim_pending_locked
    dropped: int = 0         # discarded (only on visibility-hide flush or
                             # single oversize entry, never on coalesce path)


@dataclass
class _EmitWindow:
    """Per-stats-window emit-loop telemetry. Rebound to a fresh instance
    each stats window — see Dashboard._log_stats."""
    ticks: int = 0
    calls: int = 0
    ns_sum: int = 0
    ns_max: int = 0
    batch_sum: int = 0
    batch_max: int = 0


class Dashboard:
    def __init__(
        self,
        buffer_size: int = 100,
        ofdm_symbol_size: int = 1272,
        bw: float = 38.16e6,
        center_freq: float = 3.6192e9,
        num_prbs: int = 106,
        first_carrier_offset: int = 900,
        classifier=None,
        adaptiveThreshold: bool = False,
        control: bool = False,
        label_callback=None,
        initial_label: str = "",
        port: int = 7778,
        show_controls: bool = False,
        aggregation_size: int = 14,
    ):
        # --- Flask + SocketIO ---
        # compression_threshold=0 → engineio compresses every outbound frame.
        self.app = Flask(__name__)
        self.app.config["TEMPLATES_AUTO_RELOAD"] = True
        self.socketio = SocketIO(self.app, compression_threshold=0)
        eio = self.socketio.server.eio
        dapp_logger.info(
            "[DASHBOARD] SocketIO async_mode=%s compression_threshold=%s",
            self.socketio.async_mode,
            getattr(eio, "compression_threshold", "?"),
        )
        self.app.add_url_rule("/", view_func=self.index)

        # --- Public params (read by handle_initial_connection and producers) ---
        self.buffer_size = buffer_size
        self.ofdm_symbol_size = ofdm_symbol_size
        self.bw = bw
        self.center_freq = center_freq
        self.num_prbs = num_prbs
        self.first_carrier_offset = first_carrier_offset
        self.control = control
        self.classifier = classifier
        self.adaptiveThreshold = adaptiveThreshold
        self.label_callback = label_callback
        self.current_label = initial_label
        self.port = port
        self.show_controls = show_controls

        # OAI FFT layout: bin 0 = DC, then positive freqs, then aliased
        # negative freqs at the end. We re-emit [neg | pos] in increasing
        # frequency order. _n_active is the displayed-band width in bins.
        self._n_active = self.num_prbs * 12
        self._neg_active_start = self.first_carrier_offset
        self._pos_active_end = self._n_active // 2

        # --- Display knobs ---
        # Frame-skip: 1 = no skip. Lets the user trade dashboard update rate
        # for dApp CPU. The constant-rate emitter does its own pacing.
        self.sampling_counter = 0
        self.sampling_threshold = 1
        # Row aggregation: collapse N input rows → ceil(N/agg) output rows by
        # mean. agg=14 collapses a full UL slot to one row (default); agg=1
        # keeps every per-symbol row. Tunable from sidebar.
        self._aggregation_size = max(1, int(aggregation_size))

        # --- Producer↔emit-loop shared state (all under _latest_lock) ---
        self._latest_lock = threading.Lock()
        self._pending_mag_rows = []  # list[(row_count, uint16_bytes)]
        self._latest_producer_ts_ns = 0
        self._latest_prb_list = None
        self._prb_dirty = False
        self._latest_anf = None
        self._anf_dirty = False
        # Flow control: deque of in-flight emit timestamps for the K-inflight
        # gate + watchdog.
        self._inflight_ts = deque()
        # Visibility gate: while hidden we mute emits and discard staged rows
        # so engineio's queue doesn't grow against a non-draining browser.
        self._client_visible = True

        self._rows = _RowCounters()
        self._emit = _EmitWindow()
        # Last-window cumulative submitted, used by _log_stats to detect
        # per-window producer stalls (delta == 0 with ticks > 0 = freeze).
        self._prev_submitted = 0

        # --- SocketIO event handlers ---
        self.socketio.on_event("connect", self.handle_initial_connection)
        self.socketio.on_event("disconnect", self._on_disconnect)
        if self.label_callback is not None:
            self.socketio.on_event("set_ground_truth_label", self._on_set_label)
        self.socketio.on_event("set_sampling_threshold", self._on_set_sampling_threshold)
        self.socketio.on_event("set_aggregation_size", self._on_set_aggregation_size)
        self.socketio.on_event("client_done", self._on_client_done)
        self.socketio.on_event("set_client_visible", self._on_set_client_visible)

        # --- Threads ---
        self._emit_stop_event = threading.Event()
        self.run_thread = threading.Thread(
            target=self.run, daemon=True, name="dashboard-server",
        )
        self._emit_thread = threading.Thread(
            target=self._emit_loop, daemon=True, name="dashboard-emit",
        )
        self.run_thread.start()
        self._emit_thread.start()

    # --- HTTP / lifecycle ----------------------------------------------------

    def index(self):
        return render_template("index.html")

    def run(self):
        self.socketio.run(
            self.app, host="0.0.0.0", port=self.port,
            debug=False, use_reloader=False, allow_unsafe_werkzeug=True,
        )

    def stop(self):
        self._emit_stop_event.set()
        for t in (self._emit_thread, self.run_thread):
            if t and t.is_alive():
                t.join(timeout=1)

    # --- SocketIO event handlers ---------------------------------------------

    def handle_initial_connection(self):
        # New client: clear in-flight bookkeeping (prior emits will never be
        # acked) and re-arm the visibility flag.  A fresh page-load is by
        # definition visible; don't wait for the JS-side emitVisibility()
        # follow-up to flip the flag.  Without this re-arm, a disconnect
        # that didn't fire _on_disconnect (process kill, network blip) can
        # leave the flag pinned False from a prior set_client_visible(false)
        # emitted by the visibilitychange listener.
        with self._latest_lock:
            self._inflight_ts.clear()
            self._client_visible = True
        self.socketio.emit("initialize_plot", {
            "num_bins": self._n_active,
            "waterfall_depth": self.buffer_size,
            "center_freq": self.center_freq,
            "bw": self.bw,
            "num_prbs": self.num_prbs,
            "first_carrier_offset": self.first_carrier_offset,
            # PRB-zone overlay (just the BWP-edge guard band marker) only
            # meaningful when control is active.
            "show_prb_zones": self.control,
            "predicted_label": self.classifier is not None,
            "adaptive_noise_floor": self.adaptiveThreshold,
            "show_label_selector": self.label_callback is not None,
            "current_label": self.current_label,
            "show_controls": self.show_controls,
            "sampling_threshold": self.sampling_threshold,
            "aggregation_size": self._aggregation_size,
        })

    def _on_disconnect(self):
        # Browser tab closed / refreshed / network blip.  Without this
        # reset, a trailing set_client_visible(false) emitted by the dying
        # client (visibilitychange or page-unload) would pin _client_visible
        # to False forever — the producer and emit-loop both short-circuit
        # on that flag, so rows would silently stop reaching ANY future
        # client until the dApp is restarted.  Clear in-flight stamps too
        # since prior emits will never be acked.
        with self._latest_lock:
            self._client_visible = True
            self._inflight_ts.clear()
        dapp_logger.info("[DASHBOARD] client disconnected; reset visibility + inflight")

    def _on_set_label(self, label):
        self.current_label = label
        self.label_callback(label)

    def _on_set_sampling_threshold(self, value):
        self._handle_int_setting(value, "sampling_threshold", "update_sampling_threshold")

    def _on_set_aggregation_size(self, value):
        # Int assignment is atomic on CPython; readers don't need the lock.
        self._handle_int_setting(value, "_aggregation_size", "update_aggregation_size")

    def _handle_int_setting(self, value, attr: str, broadcast_event: str):
        """Generic min=1 int setter: validate, store, broadcast the new value
        + reset_waterfall (so the canvas starts fresh under the new setting)."""
        try:
            v = int(value)
        except (TypeError, ValueError):
            return
        if v < 1:
            return
        setattr(self, attr, v)
        self.socketio.emit(broadcast_event, v)
        self.socketio.emit("reset_waterfall")

    def _on_client_done(self):
        # Browser finished painting one frame → retire the oldest in-flight ack.
        with self._latest_lock:
            if self._inflight_ts:
                self._inflight_ts.popleft()

    def _on_set_client_visible(self, visible):
        # Hidden → mute emits + drop staged rows (browser can't drain; engineio
        # queue would grow). Visible → reset deque so next emit fires fresh.
        v = bool(visible)
        with self._latest_lock:
            was = self._client_visible
            self._client_visible = v
            if not v:
                if self._pending_mag_rows:
                    self._rows.dropped += sum(c for c, _ in self._pending_mag_rows)
                    self._pending_mag_rows = []
                self._prb_dirty = False
                self._anf_dirty = False
            elif not was:
                self._inflight_ts.clear()

    def emit_label(self, label: str):
        self.current_label = label
        self.socketio.emit("update_ground_truth_label", label)

    # --- Producer path -------------------------------------------------------

    def process_iq_data(self, message):
        """Dispatch a dashboard message. Supported tags:
          * "iq_mag_batch"         - 2D (n_sym, fft_size) Float32 magnitude.
                                     Optional 3rd tuple element = producer
                                     CLOCK_MONOTONIC ns timestamp (data-age).
          * "prb_list"             - detected PRB indices (numpy array).
          * "adaptive_noise_floor" - noise floor curve from the detector.
        """
        if len(message) == 3:
            plot, payload, producer_ts_ns = message
        else:
            plot, payload = message
            producer_ts_ns = 0

        if plot == "iq_mag_batch":
            self.sampling_counter += 1
            if self.sampling_counter < self.sampling_threshold:
                return
            self.sampling_counter = 0
            self._handle_iq_mag_batch(payload, producer_ts_ns)
        elif plot == "prb_list":
            # Latest-wins state — single slot, the emit-loop reads at its
            # next tick and clears the dirty flag.
            with self._latest_lock:
                self._latest_prb_list = payload
                self._prb_dirty = True
        elif self.adaptiveThreshold and plot == "adaptive_noise_floor":
            with self._latest_lock:
                self._latest_anf = payload
                self._anf_dirty = True

    def _handle_iq_mag_batch(self, payload, producer_ts_ns):
        # Visibility short-circuit before any allocation/CPU work — keeps the
        # producer thread idle when the tab is hidden.
        if not self._client_visible:
            return
        # mag → dB in-place, slice active band, optionally decimate, aggregate
        # rows per the slider, quantize Float32→Uint16 for the wire.
        mag_db_2d = self._mag_to_db(payload)
        active_2d = self._slice_active_band_2d(mag_db_2d)
        active_2d = self._aggregate_rows(active_2d)
        n_rows = int(active_2d.shape[0])
        blob = self._quantize_to_uint16_bytes(active_2d)
        with self._latest_lock:
            self._rows.submitted += n_rows
            self._pending_mag_rows.append((n_rows, blob))
            if producer_ts_ns:
                self._latest_producer_ts_ns = producer_ts_ns
            self._trim_pending_locked()
        if self.classifier:
            # Classifier still operates per-symbol; pick the last symbol of
            # the batch (preserves prior behaviour).
            label = self.classifier.predict(payload[-1])
            self.socketio.emit("update_plot", {"predicted_label": label})

    @staticmethod
    def _mag_to_db(mag):
        """In-place 20*log10(max(mag, 1.0)). Caller owns `mag` and may mutate."""
        np.maximum(mag, 1.0, out=mag)
        np.log10(mag, out=mag)
        mag *= 20.0
        return mag

    def _slice_active_band_2d(self, mag_db_2d):
        """Extract the active band in increasing-frequency order from a 2D
        (rows, fft_size) array.

        Two layouts to handle:

        - first_carrier_offset == 0: the producer (OAI /e3_ran_buffers shm
          path, used by spectrum_dapp) already wrote PRBs in linear,
          increasing-frequency order — PRB 0 is the lowest UL freq, the
          first num_prbs*12 columns are the active band, the rest is
          zero-padded layout slack. Trim to the active band and done.

        - first_carrier_offset > 0: legacy bin-order layout where bin 0 is
          DC, positive freqs follow, negative freqs are aliased at the end
          of the FFT. Re-emit as [negative | positive] in increasing-freq
          order via the concatenate path. Preserved so a future raw
          rxdataF source still works.

        The first branch is what the spectrum dApp hits today. The old
        unconditional concatenate path was producing a phantom mirrored
        copy of the low-frequency PRBs at the right edge of the display
        (columns 0..635 reappearing at columns 2048..2683) because with
        first_carrier_offset=0 the `neg` slice covered the entire row and
        the `pos` slice duplicated its first half.
        """
        if self._neg_active_start == 0:
            return mag_db_2d[:, :self._n_active]
        neg = mag_db_2d[:, self._neg_active_start:]
        pos = mag_db_2d[:, :self._pos_active_end]
        return np.concatenate((neg, pos), axis=1)

    def _aggregate_rows(self, active_2d):
        """Collapse the row dimension by _aggregation_size via mean. agg>=n_in
        collapses everything to one row; agg=1 returns input untouched.
        In between, group into ceil(n_in/agg) chunks via reduceat (handles
        ragged tail)."""
        n_in = int(active_2d.shape[0])
        agg = self._aggregation_size
        if agg <= 1 or n_in <= 1:
            return active_2d
        if agg >= n_in:
            return active_2d.mean(axis=0, keepdims=True)
        chunk_starts = np.arange(0, n_in, agg)
        sums = np.add.reduceat(active_2d, chunk_starts, axis=0)
        chunk_ends = np.append(chunk_starts[1:], n_in)
        counts = (chunk_ends - chunk_starts).astype(np.float32)
        return sums / counts[:, None]

    @staticmethod
    def _quantize_to_uint16_bytes(active_2d):
        scaled = (active_2d - QUANT_MIN_DB) * QUANT_SCALE
        np.clip(scaled, 0.0, 65535.0, out=scaled)
        return np.ascontiguousarray(scaled.astype(np.uint16, copy=False)).tobytes()

    # --- Queue management (caller-locked) ------------------------------------

    def _trim_pending_locked(self):
        """Bound row count to MAX_PENDING_MAG_ROWS by mean-folding the oldest
        two entries (information preserved as time-average). Falls back to
        drop only if a single entry alone exceeds the cap (defensive)."""
        total = sum(c for c, _ in self._pending_mag_rows)
        while total > MAX_PENDING_MAG_ROWS:
            if len(self._pending_mag_rows) >= 2:
                self._coalesce_oldest_two_locked()
                total = sum(c for c, _ in self._pending_mag_rows)
            else:
                dropped_count, _ = self._pending_mag_rows.pop(0)
                self._rows.dropped += dropped_count
                total -= dropped_count

    def _coalesce_oldest_two_locked(self):
        """Replace the two oldest entries with a single 1-row entry that is
        the weighted-by-count mean of all their rows. Wire format is uint16;
        we widen to uint32 for the sum (no overflow), then re-quantize."""
        c_a, blob_a = self._pending_mag_rows.pop(0)
        c_b, blob_b = self._pending_mag_rows.pop(0)
        arr_a = np.frombuffer(blob_a, dtype=np.uint16).reshape(c_a, -1)
        arr_b = np.frombuffer(blob_b, dtype=np.uint16).reshape(c_b, -1)
        total_rows = c_a + c_b
        summed = arr_a.sum(axis=0, dtype=np.uint32) + arr_b.sum(axis=0, dtype=np.uint32)
        mean = (summed + total_rows // 2) // total_rows  # round to nearest
        merged = np.ascontiguousarray(mean.astype(np.uint16)).tobytes()
        self._rows.coalesced += (total_rows - 1)
        # Re-insert at the head so chronological order is preserved.
        self._pending_mag_rows.insert(0, (1, merged))

    # --- Emit loop -----------------------------------------------------------

    def _emit_loop(self):
        """Background timer: drain pending state at ~60 Hz, gated by visibility
        and K-in-flight. Stats log every 5 s."""
        interval = 1.0 / max(1.0, DEFAULT_EMIT_TARGET_FPS)
        next_deadline = time.monotonic()
        last_stats_t = time.monotonic()
        dapp_logger.info("[DASHBOARD] emit_loop TID=%d", threading.get_native_id())

        while not self._emit_stop_event.is_set():
            # Wrap the whole iteration in try/except so an unexpected
            # exception in _take_pending / _do_emit / _log_stats never
            # silently kills this daemon thread.  _do_emit has its own
            # narrow try/except around the actual socketio.emit call;
            # this outer net catches anything outside that scope (lock
            # contention bugs, time.monotonic patched out under test,
            # future refactors that move work outside _do_emit, etc.).
            # We log + back off briefly and continue — far better than
            # the dashboard going permanently dark with no signal.
            try:
                self._emit.ticks += 1
                # Snapshot pending state under the lock, emit outside it so we
                # don't serialise the producer.
                self._do_emit(*self._take_pending())

                now_t = time.monotonic()
                if now_t - last_stats_t >= STATS_INTERVAL_S:
                    self._log_stats(now_t - last_stats_t)
                    last_stats_t = now_t

                next_deadline += interval
                sleep_for = next_deadline - time.monotonic()
                if sleep_for > 0:
                    self._emit_stop_event.wait(sleep_for)
                else:
                    # Fell behind by > 1 interval (slow emit, GC pause). Rebaseline
                    # rather than burst-catching-up. Yield briefly so we don't
                    # busy-spin if emits routinely overrun the interval.
                    self._emit_stop_event.wait(0.001)
                    next_deadline = time.monotonic()
            except Exception:
                dapp_logger.exception(
                    "[DASHBOARD] emit_loop iteration failed; continuing"
                )
                # Brief back-off so a persistent fault doesn't spin the
                # CPU at 100% emitting exception traces.  The watchdog
                # in _take_pending + the per-window WARN heartbeat in
                # _log_stats remain functional through transient
                # failures.
                self._emit_stop_event.wait(0.1)
                next_deadline = time.monotonic() + interval

    def _take_pending(self):
        """Atomically (a) drop watchdog-expired in-flight entries, (b) check
        the gate, (c) drain pending rows + dirty flags. Returns the outputs
        for the unlocked emit below. Short-circuits when hidden — the
        visibility handler already cleared pending state."""
        mag_batch = None
        rows_in_batch = 0
        ts_ns = 0
        prb = None
        anf = None
        with self._latest_lock:
            if not self._client_visible:
                return mag_batch, rows_in_batch, ts_ns, prb, anf
            now_ns = time.monotonic_ns()
            while (
                self._inflight_ts
                and (now_ns - self._inflight_ts[0]) >= EMIT_WATCHDOG_NS
            ):
                self._inflight_ts.popleft()
            if self._pending_mag_rows and len(self._inflight_ts) < MAX_INFLIGHT:
                rows_in_batch = sum(c for c, _ in self._pending_mag_rows)
                mag_batch = b"".join(b for _, b in self._pending_mag_rows)
                self._pending_mag_rows = []
                ts_ns = self._latest_producer_ts_ns
                self._inflight_ts.append(now_ns)
            if self._prb_dirty:
                prb = self._latest_prb_list
                self._prb_dirty = False
            if self._anf_dirty:
                anf = self._latest_anf
                self._anf_dirty = False
        return mag_batch, rows_in_batch, ts_ns, prb, anf

    def _do_emit(self, mag_batch, rows_in_batch, ts_ns, prb, anf):
        try:
            if mag_batch is not None:
                age_us = (
                    (time.monotonic_ns() - ts_ns) / 1000.0 if ts_ns > 0 else 0.0
                )
                age_header = np.array([age_us], dtype=np.float64).tobytes()
                t0 = time.monotonic_ns()
                self.socketio.emit("update_plot_mag", age_header + mag_batch)
                elapsed = time.monotonic_ns() - t0
                # Telemetry: only counted after emit returned successfully.
                self._emit.ns_sum += elapsed
                self._emit.ns_max = max(self._emit.ns_max, elapsed)
                self._emit.calls += 1
                self._emit.batch_sum += rows_in_batch
                self._emit.batch_max = max(self._emit.batch_max, rows_in_batch)
                self._rows.emitted += rows_in_batch
            if prb is not None:
                self.socketio.emit("update_plot", {"prb_list": prb.tolist()})
            if anf is not None:
                self.socketio.emit("update_plot", {"adaptive_noise_floor": anf.tolist()})
        except Exception as exc:
            # Transient socket.io hiccup; retry next tick. Log at debug
            # so persistent failures are diagnosable but quiet hiccups
            # don't spam INFO.
            dapp_logger.debug("[DASHBOARD] emit failed: %r", exc)

    def _log_stats(self, window_s):
        """Periodic [DASHBOARD] log: cumulative row counters + per-window
        emit telemetry, then reset the per-window counters.

        Also raises a WARN when this window saw zero new submissions
        while the emit loop kept ticking — the freeze symptom — so a
        transient stall self-diagnoses without needing client-side
        instrumentation.  The diagnostic prints visible/pending/inflight
        to disambiguate the four likely causes: client visibility flag
        stuck, server-side K-in-flight saturation, upstream producer
        stall, or emit thread death."""
        r = self._rows
        e = self._emit
        self._emit = _EmitWindow()  # rebind; e still references the captured window
        window_submitted = r.submitted - self._prev_submitted
        self._prev_submitted = r.submitted
        if window_submitted == 0:
            # Two distinct cases:
            #   - warmup: no data has ever flowed (r.submitted == 0)
            #   - freeze: cumulative held steady across this window
            # Only warn for the freeze case (cumulative > 0).
            if r.submitted > 0 and e.ticks > 0:
                dapp_logger.warning(
                    "[DASHBOARD] no rows submitted in last %.1fs "
                    "(visible=%s pending=%d inflight=%d ticks=%d calls=%d) — "
                    "dashboard may be frozen; refresh the browser tab if so",
                    window_s, self._client_visible,
                    len(self._pending_mag_rows), len(self._inflight_ts),
                    e.ticks, e.calls,
                )
            return  # warmup or stuck
        dapp_logger.info(
            "[DASHBOARD] mag rows: submitted=%d emitted=%d coalesced=%d (%.2f%%) "
            "server_dropped=%d (%.2f%%)",
            r.submitted, r.emitted,
            r.coalesced, 100.0 * r.coalesced / r.submitted,
            r.dropped, 100.0 * r.dropped / r.submitted,
        )
        calls = max(1, e.calls)
        dapp_logger.info(
            "[DASHBOARD] emit_loop: tick_hz=%.1f call_hz=%.1f mean_us=%.0f "
            "max_us=%.0f mean_batch=%.1f max_batch=%d",
            e.ticks / window_s, e.calls / window_s,
            (e.ns_sum / calls) / 1000.0, e.ns_max / 1000.0,
            e.batch_sum / calls, e.batch_max,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dashboard interactive demo")
    parser.add_argument("--initial-label", default="", metavar="LABEL",
                        help="Ground truth label to pre-populate in the GUI")
    parser.add_argument("--port", type=int, default=7778)
    args = parser.parse_args()

    def on_label(label):
        print(f"[demo] ground_truth_label updated: {label!r}")

    demo = Dashboard(
        label_callback=on_label,
        initial_label=args.initial_label,
        port=args.port,
    )
    print(f"Dashboard running at http://localhost:{args.port}")
    print("Type a label and press Enter to push it to the GUI, or Ctrl-C to quit.")
    try:
        while True:
            label = input("label> ").strip()
            if label:
                demo.emit_label(label)
                print(f"  → pushed {label!r} to browser")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        demo.stop()
