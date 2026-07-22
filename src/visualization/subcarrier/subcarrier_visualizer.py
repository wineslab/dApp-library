#!/usr/bin/env python3

#
# SPDX-FileCopyrightText: Copyright (c) 2026 Northeastern University. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""
Subcarrier Power Visualizer.

Receives per-antenna per-subcarrier power blobs from the subcarrier-power
dApp on a ZMQ SUB socket, fans them out to connected browsers over a
WebSocket as binary frames, and renders four stacked waterfalls in WebGL2
(one per antenna, full subcarrier resolution: 273 PRBs * 12 SCs = 3276 cols).

Wire format on the WebSocket (binary frame):
    [12B header: u16 sfn, u16 slot, u16 n_ant, u16 n_sym, u16 n_sc, i8 db_min, i8 db_max]
    [n_ant * n_sym * n_sc * u8 power values]
    [u8 det_present]
        if det_present == 1:
            [u16 det_n LE][det_n * u8 blocked mask]
"""

import argparse
import json
import os
import queue
import struct
import subprocess
import threading
import time
import zlib

import numpy as np
import zmq
from flask import Flask, jsonify, render_template_string, request, send_file
from flask_cors import CORS
from flask_sock import Sock

# Disable permessage-deflate. simple_websocket hard-enables it on every WS
# (ws.py: AcceptConnection(extensions=[PerMessageDeflate()])), but the dApp's IQ
# power frames are essentially incompressible, so zlib-compressing them buys ~no
# wire savings while burning GIL time. With N parallel streams that wasted
# compression serializes the sender threads AND starves the ZMQ receiver (which
# then back-pressures the dApp's PUB). Declining the client's offer (accept ->
# None) negotiates the connection WITHOUT deflate: frames go out uncompressed, so
# each sender spends its time blocked on its own socket (GIL released) and the N
# streams genuinely run in parallel, while the receiver keeps draining at line rate.
try:
    from wsproto.extensions import PerMessageDeflate as _PerMessageDeflate
    _PerMessageDeflate.accept = lambda self, offer: None
except Exception as _e:  # pragma: no cover - never fatal; deflate stays on if the API moved
    print(f"[WARN] could not disable permessage-deflate: {_e}")

app = Flask(__name__)
CORS(app)
sock = Sock(app)

# Number of parallel WebSocket streams the browser opens (frame-interleave). A
# single WS is TCP-window limited (~70 Mbit/s here regardless of link speed); N
# parallel streams each get their own window so aggregate throughput scales ~N×.
# Injected into the page as __N_STREAMS__; override with the VIZ_STREAMS env var.
N_STREAMS = int(os.environ.get('VIZ_STREAMS', '4'))

# --- Optional WS wire compression (off by default; A/B via VIZ_COMPRESS) ------
# Generic compression of the raw u8 power is useless (~1.3x) because the noise
# floor jitters per sample (see the permessage-deflate note above). But if we
# (a) collapse the sub-(floor+delta) noise to one level and (b) round to 2^BITS
# colormap levels, the frame becomes highly compressible and a plain deflate of
# each batch hits ~12x on real frames (~290 -> ~23 Mbit/s) at FULL spectral
# resolution. The transform is lossless for the SIGNAL (anything above the floor)
# and display-equivalent (a turbo colormap needs no 256 levels). The browser
# inflates with the native DecompressionStream (no JS/wasm dependency). All gated
# by VIZ_COMPRESS and injected into the page as __COMPRESS__ so it is A/B-able.
VIZ_COMPRESS      = os.environ.get('VIZ_COMPRESS', '1') == '1'   # default ON; set VIZ_COMPRESS=0 to disable
VIZ_COMPRESS_BITS = max(1, min(8, int(os.environ.get('VIZ_COMPRESS_BITS', '8'))))   # colormap levels = 2^BITS (8 = full 256; floor-clamp does the compressing)
VIZ_FLOOR_DELTA   = max(0, int(os.environ.get('VIZ_FLOOR_DELTA', '8')))             # u8 steps above floor to keep (0 = off)
VIZ_ZLIB_LEVEL    = max(1, min(9, int(os.environ.get('VIZ_ZLIB_LEVEL', '6'))))

# Full-resolution grid width. The dApp publishes num_prbs*12 subcarrier columns
# (e.g. 106 PRB -> 1272), so the visualizer must size its WebGL resources and the
# frame-acceptance gate to the *active* bandwidth, not a fixed 273-PRB grid — a
# mismatch makes every frame fail the N_SC%nSc divisor gate and nothing renders.
# Injected into the page as __N_SC__/__N_PRBS__; driven by --num-prbs.
VIZ_SC_PER_PRB = 12
VIZ_NUM_PRBS   = int(os.environ.get('VIZ_NUM_PRBS', '273'))

# --- End-to-end pacing (browser-feedback shedding; default ON) ----------------
# A proxy/ingress between the viz and the browser hides the slow hop from the
# viz's own socket, so socket-level back-pressure can't see it. Instead we pace
# on the browser's OWN feedback: it reports cumulative `consumed` via /clientstats
# (tagged with a per-page-load `sid`), and we shed a client's batch whenever the
# in-flight (frames pushed - frames consumed) exceeds VIZ_PACE_BUDGET. That bounds
# the end-to-end backlog THROUGH any proxy, so SC mode stays real-time (drops
# frames rather than buffering). Independent of compression.
VIZ_PACE        = os.environ.get('VIZ_PACE', '1') == '1'
VIZ_PACE_BUDGET = max(16, int(os.environ.get('VIZ_PACE_BUDGET', '96')))    # max in-flight frames/browser; pacing DRAINS to this (~0.3s lag at SC rates)


def _preprocess_power(raw):
    """Floor-clamp + bit-depth-quantize the u8 power so it compresses well. Keeps
    the u8 wire format (the browser parses it unchanged) -- only the VALUES
    change: sub-(median+delta) noise collapses to the floor and values round to
    2^BITS levels. Returns bytes(); lossless above the floor, display-equivalent."""
    a = np.frombuffer(raw, dtype=np.uint8).copy()
    if VIZ_FLOOR_DELTA > 0:
        floor = int(np.median(a))
        a[a < floor + VIZ_FLOOR_DELTA] = floor
    if VIZ_COMPRESS_BITS < 8:
        sh = 8 - VIZ_COMPRESS_BITS
        np.right_shift(a, sh, out=a)
        np.left_shift(a, sh, out=a)
    return a.tobytes()

# Per-client outbound state. Each connected browser gets a WsClient: an outbound
# queue its own ws_route thread drains serially (so a Sock is never shared across
# threads), plus a frame-interleave assignment (stream k of K). The zmq_receiver
# owns each client's `pending` batch list and routes only the frames in that
# client's bucket — idx % K == k — so K parallel WS connections each carry 1/K of
# the frames over their OWN TCP stream. A single WebSocket is window-limited by
# the bandwidth-delay product (~70 Mbit/s here regardless of the 1 Gb/s link);
# K parallel streams each get their own window, so aggregate scales ~K×. K=1
# (default / no query) => every frame to every client (original behavior).
class WsClient:
    __slots__ = ('q', 'k', 'K', 'pending', 'sid')

    def __init__(self, q, k, K, sid=''):
        self.q = q
        self.k = k
        self.K = K
        self.sid = sid      # links this browser's K streams to its /clientstats (pacing)
        self.pending = []   # accumulating batch (owned by the zmq_receiver thread)


_ws_clients = []
_ws_lock = threading.Lock()

# Frames-pushed-to-WS vs frames-received diagnostics. _ws_sent counts payloads
# actually ws.send()'d to a client (i.e. consumed by the browser's TCP), _ws_shed
# counts payloads dropped in _enqueue because a client queue was full (browser /
# transport behind). Compared against the ZMQ receive count, a growing shed / a
# ws-push rate below the recv rate means the bottleneck is the WS transfer +
# browser render, not the dApp pipeline. Lock order: _ws_lock -> _ws_stat_lock.
_ws_stat_lock = threading.Lock()
_ws_sent = 0
_ws_shed = 0

# End-to-end pacing state, keyed by per-page-load `sid` (a browser = its K WS
# streams sharing one sid). _pace_sent = cumulative frames enqueued toward that
# browser (summed across its streams); _pace_consumed = the latest cumulative
# `consumed` the browser reported via /clientstats. in-flight = sent - consumed.
_pace_lock = threading.Lock()
_pace_sent = {}
_pace_consumed = {}
_pace_consumed_base = {}   # rebase: browser `consumed` is cumulative since page load, but
                           # _pace_sent restarts with the server; align them at the sid's
                           # first beacon so sent/consumed share a baseline (survives a viz
                           # restart without a browser reload).
_pace_paused = {}          # per-sid paused/hidden flag: stop sending + re-baseline on resume
                           # (a hidden tab stops consuming, so pacing would otherwise pin the
                           # backlog at the budget forever -> frozen feed until a refresh).


def _enqueue(q, payload_bytes, n_frames):
    # Hand one batched WS message (carrying n_frames frames) to a client queue.
    # On overflow (browser/transport behind) drop the OLDEST batch, count its
    # frames as shed, then retry once. shed/sent are in FRAMES (not batches) so
    # "ws push Hz" stays comparable to the receive rate.
    global _ws_shed
    item = (payload_bytes, n_frames)
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            old = q.get_nowait()
            with _ws_stat_lock:
                _ws_shed += old[1]
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            with _ws_stat_lock:
                _ws_shed += n_frames


@sock.route('/ws')
def ws_route(ws):
    global _ws_sent
    # Frame-interleave assignment from the query: ?stream=k&of=K. The browser
    # opens K connections (k=0..K-1); this client receives only frames whose
    # index falls in bucket k. Defaults to the full stream (k=0, K=1).
    try:
        K = max(1, int(request.args.get('of', '1')))
        k = int(request.args.get('stream', '0')) % K
    except (ValueError, TypeError):
        k, K = 0, 1
    sid = request.args.get('sid', '')
    q = queue.Queue(maxsize=4)   # small: bound buffered latency to a fraction of a second so the
                                 # waterfall is real-time; overflow drops the OLDEST (shed) rather than
                                 # holding ~seconds of backlog. (Was 128 -> up to ~7s of delay when behind.)
    client = WsClient(q, k, K, sid)
    with _ws_lock:
        _ws_clients.append(client)
    try:
        while True:
            try:
                item = q.get(timeout=15.0)
            except queue.Empty:
                # Idle: probe the socket so a half-closed connection breaks out.
                try:
                    ws.receive(timeout=0)
                except Exception:
                    break
                continue
            payload, nf = item
            try:
                ws.send(payload)
            except Exception:
                break
            with _ws_stat_lock:
                _ws_sent += nf
    finally:
        with _ws_lock:
            if client in _ws_clients:
                _ws_clients.remove(client)
            sid_gone = bool(sid) and not any(cc.sid == sid for cc in _ws_clients)
        if sid_gone:
            with _pace_lock:
                _pace_sent.pop(sid, None)
                _pace_consumed.pop(sid, None)
                _pace_consumed_base.pop(sid, None)
                _pace_paused.pop(sid, None)


def zmq_receiver(zmq_port):
    ctx = zmq.Context()
    s = ctx.socket(zmq.SUB)
    s.setsockopt(zmq.RCVHWM, 8)   # don't buffer a stale backlog at the SUB; drain-to-latest below keeps us current
    s.connect(f"tcp://localhost:{zmq_port}")
    s.setsockopt_string(zmq.SUBSCRIBE, "")
    s.setsockopt(zmq.RCVTIMEO, 1000)

    print(f"ZMQ receiver connected to localhost:{zmq_port}")
    print("Waiting for subcarrier_power frames...")

    total_count = 0
    window_count = 0
    last_log = time.monotonic()
    last_ws_sent = 0
    last_ws_shed = 0

    # Frame batching (perf #1): coalesce all frames arriving within BATCH_MS into
    # ONE WebSocket message PER CLIENT — [u16 count] + concatenated self-describing
    # per-frame blobs (each = the same 14B header + raw + trailer as before). This
    # cuts the WS message rate ~13x while preserving every frame. With frame-
    # interleave (K>1) each client's batch holds only its bucket's frames, so K
    # clients => K parallel WS streams each carrying 1/K of the data; the browser
    # merges the whole frames back in arrival order.
    BATCH_MS = 0.016   # ~60 Hz flush cadence (matches display refresh)
    BATCH_MAX = 64     # cap frames/batch to bound latency

    clients = []        # window snapshot of WsClient (refreshed each flush)

    def _refresh_clients():
        nonlocal clients
        with _ws_lock:
            clients = list(_ws_clients)

    _refresh_clients()
    win_start = time.monotonic()

    def _flush():
        nonlocal win_start
        global _ws_shed
        for c in clients:
            if c.pending:
                nf = len(c.pending)
                # End-to-end pacing: if this browser is behind (in-flight beyond
                # budget) shed this batch instead of enqueueing it -> bounds the
                # backlog through any proxy. No-op until the browser has reported.
                if VIZ_PACE and c.sid:
                    with _pace_lock:
                        paused_cli = _pace_paused.get(c.sid, False)
                        sent = _pace_sent.get(c.sid, 0)
                        backlog = sent - _pace_consumed.get(c.sid, sent)
                    # Paused/hidden tab: don't send (and don't grow `sent`) -> saves bandwidth
                    # and avoids a stuck backlog; the resume beacon re-baselines (above).
                    if paused_cli or backlog > VIZ_PACE_BUDGET:
                        with _ws_stat_lock:
                            _ws_shed += nf
                        c.pending = []
                        continue
                msg = struct.pack('<H', nf) + b''.join(c.pending)
                if VIZ_COMPRESS:
                    msg = zlib.compress(msg, VIZ_ZLIB_LEVEL)
                _enqueue(c.q, msg, nf)
                if VIZ_PACE and c.sid:
                    with _pace_lock:
                        _pace_sent[c.sid] = _pace_sent.get(c.sid, 0) + nf
                c.pending = []
        win_start = time.monotonic()
        _refresh_clients()   # pick up connects/disconnects for the next window

    while True:
        try:
            # recv_multipart is the only safe way to read variable-frame ZMQ
            # messages: the dApp may emit 2 frames (no detector) or 3 frames
            # (detector enabled and ready → trailing mask).
            parts_raw = s.recv_multipart()
        except zmq.Again:
            _flush()   # no new data within the timeout: flush any pending batches
            continue
        except Exception as e:
            print(f"[ERROR] ZMQ recv: {e}")
            continue
        # Real-time: drain any further queued frames, keeping only the NEWEST, so we
        # never work through a stale ZMQ backlog when per-frame preprocessing + per-
        # batch deflate can't match the dApp's PUB rate (SC mode). Without this the
        # ZMQ pipe buffers seconds of old frames -> the whole dashboard lags. This is
        # the upstream analogue of the WS pacing.
        while True:
            try:
                parts_raw = s.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                break

        if len(parts_raw) < 2:
            print(f"[ERROR] Unexpected ZMQ multipart length: {len(parts_raw)}")
            continue

        meta = parts_raw[0].decode('utf-8', errors='replace')
        raw  = bytes(parts_raw[1])
        mask = bytes(parts_raw[2]) if len(parts_raw) >= 3 else b''

        parts = meta.split('|')
        if len(parts) < 9 or parts[0] != 'subcarrier_power':
            print(f"[ERROR] Unexpected metadata: {meta}")
            continue
        try:
            sfn    = int(parts[1])
            slot   = int(parts[2])
            n_ant  = int(parts[3])
            n_sym  = int(parts[4])
            n_sc   = int(parts[5])
            dtype  = parts[6]
            db_min = int(parts[7])
            db_max = int(parts[8])
        except ValueError:
            continue

        # Optional trailing detector + perf fields (key=value, in any order).
        det_present_meta = 0
        det_n   = 0
        det_thr = 0.0
        kernel_us = 0
        for tok in parts[9:]:
            if   tok.startswith('det='):       det_present_meta = int(tok[4:])
            elif tok.startswith('det_n='):     det_n  = int(tok[6:])
            elif tok.startswith('det_thr='):   det_thr = float(tok[8:])
            elif tok.startswith('kernel_us='): kernel_us = int(tok[10:])

        if dtype != 'u8':
            print(f"[ERROR] Unsupported dtype: {dtype} (expected u8)")
            continue

        expected = n_ant * n_sym * n_sc  # 1 byte per sample
        if len(raw) != expected:
            print(f"[ERROR] Size mismatch: got {len(raw)} bytes, expected {expected}")
            continue

        # Reconcile the metadata `det=` flag with the actual presence of a
        # trailing mask frame. We only forward a mask if both agree and the
        # length matches det_n — otherwise drop it to avoid misalignment.
        det_present_wire = 1 if (det_present_meta == 1 and len(mask) == det_n and det_n > 0) else 0

        # 14-byte header: 5x u16 + 2x i8 + u16 kernel_us (clamped to 65535).
        header = struct.pack('<HHHHHbbH',
                             sfn & 0xFFFF, slot & 0xFFFF,
                             n_ant, n_sym, n_sc,
                             db_min, db_max,
                             min(kernel_us, 65535))
        if det_present_wire:
            trailer = struct.pack('<BH', 1, det_n) + mask
        else:
            trailer = struct.pack('<B', 0)
        raw_out = _preprocess_power(raw) if VIZ_COMPRESS else raw
        blob = header + raw_out + trailer
        # Frame-interleave index (μ=1: 20 slots/frame). Built once, appended to
        # whichever client(s) own this bucket. For K=1 every client gets it.
        idx = sfn * 20 + slot
        maxp = 0
        for c in clients:
            if c.K <= 1 or (idx % c.K) == c.k:
                c.pending.append(blob)
                lp = len(c.pending)
                if lp > maxp:
                    maxp = lp
        if (time.monotonic() - win_start) >= BATCH_MS or maxp >= BATCH_MAX:
            _flush()

        total_count += 1
        window_count += 1
        now = time.monotonic()
        if total_count == 1 or (now - last_log) >= 1.0:
            arr = np.frombuffer(raw, dtype=np.uint8)
            span = float(db_max - db_min)
            mx_db = (float(arr.max())  / 255.0) * span + db_min
            mn_db = (float(arr.mean()) / 255.0) * span + db_min
            with _ws_lock:
                n_clients = len(_ws_clients)
            with _ws_stat_lock:
                cur_sent = _ws_sent
                cur_shed = _ws_shed
            dt = max(now - last_log, 1e-9)
            rate = window_count / dt if total_count > 1 else 0.0
            ws_rate = (cur_sent - last_ws_sent) / dt if total_count > 1 else 0.0
            shed_win = cur_shed - last_ws_shed
            tag = "first" if total_count == 1 else f"{rate:.1f} Hz"
            with _pace_lock:
                max_backlog = max((s - _pace_consumed.get(sd, s) for sd, s in _pace_sent.items()), default=0)
            print(f"[RX] {tag} | SFN/Slot {sfn}/{slot} | n={total_count} | "
                  f"max {mx_db:.1f} dBFS | mean {mn_db:.1f} dBFS | ws clients: {n_clients} | "
                  f"ws push {ws_rate:.1f} Hz, shed {shed_win}/win | backlog {max_backlog}")
            window_count = 0
            last_log = now
            last_ws_sent = cur_sent
            last_ws_shed = cur_shed


HTML_TEMPLATE = r'''
<!DOCTYPE html>
<html>
<head>
    <title>Subcarrier Power Waterfall</title>
    <style>
        :root {
            --bg: #0e1116;
            --panel: #161b22;
            --panel-2: #0b0e13;
            --border: #30363d;
            --text: #c9d1d9;
            --muted: #8b949e;
            --accent: #58a6ff;
            --green: #3fb950;
            --yellow: #d29922;
            --red: #f85149;
            --grey: #6e7681;
        }
        * { box-sizing: border-box; }
        html, body {
            margin: 0; padding: 0; background: var(--bg); color: var(--text);
            font: 13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                  "Helvetica Neue", Arial, sans-serif;
            height: 100vh; overflow: hidden;
        }

        /* App shell: topbar | main (sidebar+dashboard) | stats-footer */
        .app {
            display: grid;
            grid-template-rows: auto 1fr auto;
            height: 100vh;
            min-height: 0;
        }
        .topbar {
            display: flex; align-items: center; gap: 10px;
            padding: 8px 14px;
            background: var(--panel); border-bottom: 1px solid var(--border);
            flex-wrap: wrap;
        }
        .topbar h1  { margin: 0; font-size: 14px; font-weight: 600; }
        .topbar p   { margin: 0; color: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
        .topbar .spacer { flex: 1; }
        .topbar .logo { height: 32px; max-width: 160px; object-fit: contain; }
        .topbar .status-pill {
            display: inline-flex; align-items: center; gap: 8px;
            padding: 4px 10px; font-size: 11px;
            background: var(--bg); border: 1px solid var(--border); border-radius: 999px;
        }
        .sidebar-toggle {
            background: var(--bg); color: var(--text);
            border: 1px solid var(--border); border-radius: 4px;
            padding: 4px 10px; font: inherit; font-size: 12px; cursor: pointer;
        }
        .sidebar-toggle:hover { border-color: var(--accent); color: var(--accent); }

        /* Main grid: sidebar | dashboard. Sidebar width toggles via root class. */
        .app__main {
            display: grid;
            grid-template-columns: 340px 1fr;
            min-height: 0; overflow: hidden;
            transition: grid-template-columns 180ms ease;
        }
        body.sidebar-closed .app__main { grid-template-columns: 0 1fr; }

        .sidebar {
            background: var(--panel); border-right: 1px solid var(--border);
            overflow-y: auto; min-height: 0;
            display: flex; flex-direction: column;
        }
        body.sidebar-closed .sidebar { display: none; }

        .sm-group { border-bottom: 1px solid var(--border); }
        .sm-group > summary {
            list-style: none; cursor: pointer; user-select: none;
            padding: 10px 14px;
            display: flex; align-items: center; gap: 8px;
            font-size: 11px; font-weight: 600; color: var(--text);
            text-transform: uppercase; letter-spacing: 0.06em;
        }
        .sm-group > summary::-webkit-details-marker { display: none; }
        .sm-group > summary::before {
            content: '\25B8'; color: var(--muted); transition: transform 120ms ease;
        }
        .sm-group[open] > summary::before { transform: rotate(90deg); }
        .sm-group > summary .sm-sub {
            color: var(--muted); font-size: 10px; font-weight: 400;
            text-transform: none; letter-spacing: 0; margin-left: auto;
        }
        .sm-body {
            padding: 4px 14px 12px;
            display: flex; flex-direction: column; gap: 6px;
            background: var(--panel-2);
            border-top: 1px solid var(--border);
        }
        .sm-row {
            display: flex; align-items: center; gap: 8px;
            font-size: 12px; flex-wrap: wrap;
        }
        .sm-row label {
            display: inline-flex; align-items: center; gap: 6px;
            min-width: 140px; flex-shrink: 0;
        }
        .sm-row input[type=number],
        .sm-row input[type=text],
        .sm-row select {
            padding: 4px 8px; background: var(--bg); color: var(--text);
            border: 1px solid var(--border); border-radius: 4px;
            font: inherit; font-variant-numeric: tabular-nums;
        }
        .sm-row input[type=number] { width: 80px; }
        .sm-row input[type=text]   { flex: 1 1 0; min-width: 60px; }
        .sm-row > .sm-help         { white-space: nowrap; }
        .sm-row input:focus, .sm-row select:focus {
            outline: 1px solid var(--accent); border-color: var(--accent);
        }
        .sm-help { color: var(--muted); font-size: 10px; }
        .sm-info {
            color: var(--muted); font-size: 11px;
            font-variant-numeric: tabular-nums; word-break: break-all;
        }
        .sm-info.ok    { color: var(--green); }
        .sm-info.error { color: var(--red); }
        .sm-actions { display: flex; align-items: center; gap: 8px; margin-top: 6px; }
        /* Free-text notes row (recording metadata): full-width textarea below label. */
        .sm-row-notes { flex-direction: column; align-items: stretch; gap: 4px; }
        .sm-row-notes > label { min-width: 0; }
        .sm-row-notes textarea {
            width: 100%; padding: 6px 8px; background: var(--bg); color: var(--text);
            border: 1px solid var(--border); border-radius: 4px;
            font: inherit; font-size: 12px; resize: vertical; min-height: 60px;
        }
        .sm-row-notes textarea:focus { outline: 1px solid var(--accent); border-color: var(--accent); }
        /* Hide the spinner arrows on <input type="number"> globally. */
        .sm-row input[type=number]::-webkit-inner-spin-button,
        .sm-row input[type=number]::-webkit-outer-spin-button {
            -webkit-appearance: none; margin: 0;
        }
        .sm-row input[type=number] { -moz-appearance: textfield; appearance: textfield; }

        /* Dashboard: vertical flex; density compact, waterfall fills remaining space. */
        .dashboard {
            display: flex; flex-direction: column;
            min-width: 0; min-height: 0;
            padding: 8px; gap: 8px; overflow: hidden;
        }
        .dashboard .plot-box {
            display: flex; flex-direction: column;
            min-height: 0; overflow: hidden;
            margin: 0;
        }
        .dashboard .plot-box.density-box   { flex: 0 0 220px; }
        .dashboard .plot-box.waterfall-box { flex: 1 1 auto; min-height: 0; }
        /* Override the legacy fixed-height .spec-wrap rules with a more specific selector. */
        .dashboard .plot-box .spec-wrap.density,
        .dashboard .plot-box .spec-wrap.waterfall {
            flex: 1 1 auto; min-height: 0; height: auto;
        }

        /* Stats footer (fixed at bottom of viewport, never scrolls). */
        .stats-footer {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 8px;
            padding: 8px 12px;
            background: var(--panel); border-top: 1px solid var(--border);
        }
        .stats-footer .stat-card {
            background: var(--panel-2); border: 1px solid var(--border);
            border-radius: 4px; padding: 6px 10px;
            display: flex; align-items: baseline; gap: 8px;
        }
        .stats-footer .stat-card h3 {
            margin: 0; font-size: 10px; font-weight: 600; color: var(--muted);
            text-transform: uppercase; letter-spacing: 0.06em;
            white-space: nowrap;
        }
        .stats-footer .stat-card p {
            margin: 0; font-size: 14px; font-weight: 600; color: var(--text);
            font-variant-numeric: tabular-nums;
        }
        .stats-footer .stat-card .unit {
            font-size: 10px; color: var(--muted); margin-left: 3px; font-weight: 400;
        }
        .stats-footer .copyright {
            grid-column: 1 / -1; text-align: center; color: var(--muted); font-size: 10px;
            padding-top: 4px; border-top: 1px solid var(--border); margin-top: 2px;
        }

        /* Two-column top bar: header + status on the left, logo cell on the right
           spanning both rows. Falls back to single column at very narrow widths. */
        .topbar-grid {
            display: grid;
            grid-template-columns: 1fr auto;
            grid-template-rows: auto auto;
            column-gap: 8px; row-gap: 8px;
            margin-bottom: 8px;
        }
        .topbar-grid .header     { grid-column: 1; grid-row: 1; margin-bottom: 0; }
        .topbar-grid .status-bar { grid-column: 1; grid-row: 2; margin-bottom: 0; }
        .logo-cell {
            grid-column: 2; grid-row: 1 / span 2;
            background: var(--panel); border: 1px solid var(--border);
            border-radius: 6px;
            padding: 10px 18px;
            display: flex; align-items: center; justify-content: center;
        }
        .logo-cell img {
            display: block; height: auto; width: auto;
            max-height: 64px; max-width: 240px;
            object-fit: contain;
        }
        @media (max-width: 700px) {
            .topbar-grid { grid-template-columns: 1fr; }
            .logo-cell   { grid-column: 1; grid-row: auto; }
        }

        .header {
            display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
            padding: 10px 14px; margin-bottom: 8px;
            background: var(--panel); border: 1px solid var(--border);
            border-radius: 6px;
        }
        .header h1 { margin: 0; font-size: 14px; font-weight: 600; color: var(--text); }
        .header p  { margin: 0; color: var(--muted); font-size: 12px;
                     font-variant-numeric: tabular-nums; }

        .status-bar {
            display: flex; align-items: center; gap: 8px;
            padding: 8px 14px; margin-bottom: 8px;
            background: var(--panel); border: 1px solid var(--border);
            border-radius: 6px; font-size: 12px;
        }
        .status-indicator {
            display: inline-block; width: 8px; height: 8px; border-radius: 50%;
            background: var(--grey); flex-shrink: 0;
        }
        .status-active  { background: var(--green); box-shadow: 0 0 6px rgba(63,185,80,0.6); }
        .status-stale   { background: var(--yellow); }
        .status-waiting { background: var(--grey); }
        .status-error   { background: var(--red); }

        .rec-bar, .viz-bar {
            display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
            padding: 8px 14px; margin-bottom: 8px;
            background: var(--panel); border: 1px solid var(--border);
            border-radius: 6px; font-size: 12px;
        }
        .rec-bar .rec-label, .viz-bar .ctrl-label {
            color: var(--muted); font-weight: 600; font-size: 11px;
            text-transform: uppercase; letter-spacing: 0.06em;
        }
        .rec-bar input[type=number], .viz-bar input[type=number] {
            width: 80px; padding: 4px 8px;
            background: var(--bg); color: var(--text);
            border: 1px solid var(--border); border-radius: 4px;
            font: inherit; font-variant-numeric: tabular-nums;
        }
        .rec-bar input[type=number]:focus, .viz-bar input[type=number]:focus {
            outline: 1px solid var(--accent); border-color: var(--accent);
        }
        .rec-bar .rec-unit, .viz-bar .ctrl-unit { color: var(--muted); font-size: 11px; }
        .viz-bar .ctrl-help {
            color: var(--muted); font-size: 11px; flex: 1; min-width: 0;
            font-variant-numeric: tabular-nums;
        }
        .rec-btn {
            padding: 4px 14px;
            background: rgba(63,185,80,0.1); color: var(--green);
            border: 1px solid var(--green); border-radius: 4px;
            font: inherit; font-weight: 600; cursor: pointer; font-size: 12px;
        }
        .rec-btn:hover { background: rgba(63,185,80,0.18); }
        .rec-btn.rec-active { color: var(--red); border-color: var(--red);
                              background: rgba(248,81,73,0.1); }
        .rec-btn.rec-active:hover { background: rgba(248,81,73,0.18); }
        .rec-btn:disabled { color: var(--muted); border-color: var(--border);
                            background: transparent; cursor: not-allowed; }
        #rec-info {
            color: var(--muted); font-size: 11px;
            flex: 1; min-width: 0; word-break: break-all;
            font-variant-numeric: tabular-nums;
        }

        .sensing-panel {
            background: var(--panel); border: 1px solid var(--border);
            border-radius: 6px; margin-bottom: 8px; overflow: hidden;
        }
        .sensing-panel > summary {
            list-style: none; cursor: pointer; user-select: none;
            padding: 8px 14px;
            display: flex; align-items: center; gap: 10px;
            font-size: 12px;
        }
        .sensing-panel > summary::-webkit-details-marker { display: none; }
        .sensing-panel > summary::before {
            content: '\25B8'; color: var(--muted); display: inline-block;
            transition: transform 120ms ease;
        }
        .sensing-panel[open] > summary::before { transform: rotate(90deg); }
        .sensing-title {
            font-weight: 600; color: var(--text);
            text-transform: uppercase; letter-spacing: 0.06em; font-size: 11px;
        }
        .sensing-sub { color: var(--muted); font-size: 11px; }
        .sensing-body {
            padding: 4px 14px 12px;
            border-top: 1px solid var(--border);
            display: flex; flex-direction: column; gap: 6px;
        }
        .sensing-row {
            display: flex; align-items: center; gap: 10px;
            padding: 4px 0; font-size: 12px;
        }
        .sensing-row label {
            display: inline-flex; align-items: center; gap: 6px;
            min-width: 170px;
        }
        .sensing-row input[type=number],
        .sensing-row input[type=text],
        .sensing-row select {
            padding: 4px 8px; background: var(--bg); color: var(--text);
            border: 1px solid var(--border); border-radius: 4px;
            font: inherit; font-variant-numeric: tabular-nums;
        }
        .sensing-row input[type=number] { width: 90px; }
        .sensing-row input[type=text]   { width: 220px; }
        .sensing-row input:focus, .sensing-row select:focus {
            outline: 1px solid var(--accent); border-color: var(--accent);
        }
        .sensing-help { color: var(--muted); font-size: 11px; }
        .sensing-actions {
            display: flex; align-items: center; gap: 10px;
            margin-top: 4px;
        }
        .sensing-info {
            color: var(--muted); font-size: 11px;
            font-variant-numeric: tabular-nums; word-break: break-all;
        }
        .sensing-info.ok    { color: var(--green); }
        .sensing-info.error { color: var(--red); }

        .plot-box {
            background: var(--panel); border: 1px solid var(--border);
            border-radius: 6px; margin-bottom: 8px; overflow: hidden;
        }
        .plot-box h2 {
            margin: 0; padding: 8px 14px;
            border-bottom: 1px solid var(--border);
            font-size: 11px; font-weight: 600; color: var(--muted);
            text-transform: uppercase; letter-spacing: 0.06em;
        }

        /* Shared layout: yaxis (80px) | canvas | colorbar | density flush over waterfall */
        .spec-wrap { position: relative; display: flex; align-items: stretch; }
        .spec-wrap.density   { height: 250px; }
        .spec-wrap.waterfall { height: 1000px; }
        .spec-yaxis {
            width: 80px; flex-shrink: 0; user-select: none;
            color: var(--muted); font-size: 11px;
            padding: 6px 8px 6px 0; text-align: right;
            display: flex; flex-direction: column; justify-content: space-between;
            font-variant-numeric: tabular-nums;
        }
        .spec-yaxis.time { align-items: center; justify-content: center; padding: 0; }
        .spec-yaxis.time .v-text {
            writing-mode: vertical-rl; transform: rotate(180deg);
            color: var(--muted); font-size: 11px; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.08em;
        }
        canvas.spec-canvas {
            flex: 1; min-width: 0; height: 100%;
            background: #000; display: block;
        }
        .spec-colorbar {
            width: 50px; padding: 4px 0; margin-left: 8px;
            display: flex; flex-direction: column; align-items: center;
            color: var(--muted); font-size: 10px; user-select: none;
            text-transform: uppercase; letter-spacing: 0.04em;
            font-variant-numeric: tabular-nums;
        }
        .spec-colorbar .cb-tick { line-height: 1; padding: 2px 0; white-space: nowrap; }
        .spec-colorbar .cb-gradient {
            flex: 1; width: 14px;
            background: linear-gradient(to top,
                #30123b 0%, #4145ab 14%, #4675ed 28%, #25d2c4 42%,
                #5cf06b 56%, #b5e043 70%, #f5a72f 84%, #cb2902 100%);
            border-radius: 2px;
        }
        .spec-xaxis {
            text-align: center; padding: 8px 14px 10px;
            color: var(--muted); font-size: 11px; user-select: none;
            text-transform: uppercase; letter-spacing: 0.04em;
            font-variant-numeric: tabular-nums;
        }
        /* Detector blocked-mask strip: aligns horizontally with the waterfall
         * canvas (matching yaxis + colorbar widths so the bins line up). */
        .mask-strip {
            height: 18px; min-height: 18px; flex: 0 0 18px;
            margin-bottom: 2px;
            background: var(--panel-2);
            border: 1px solid var(--border); border-radius: 2px;
        }
        .mask-strip .mask-label {
            width: 80px; flex-shrink: 0; padding: 0 8px 0 0;
            display: flex; align-items: center; justify-content: flex-end;
            color: var(--muted); font-size: 10px;
            text-transform: uppercase; letter-spacing: 0.06em;
        }
        .mask-strip canvas {
            flex: 1; min-width: 0; height: 100%; display: block;
        }
        .mask-strip .mask-spacer {
            width: 58px; flex-shrink: 0;
        }

        .stats-container {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 8px; margin-top: 8px;
        }
        .stat-card {
            background: var(--panel); border: 1px solid var(--border);
            border-radius: 6px; padding: 10px 14px; text-align: left;
        }
        .stat-card h3 {
            margin: 0 0 6px 0; font-size: 11px; font-weight: 600;
            color: var(--muted);
            text-transform: uppercase; letter-spacing: 0.06em;
        }
        .stat-card p {
            margin: 0; font-size: 18px; font-weight: 600; color: var(--text);
            font-variant-numeric: tabular-nums;
        }
        .stat-card .unit {
            font-size: 11px; color: var(--muted); margin-left: 4px;
            font-weight: 400; text-transform: lowercase; letter-spacing: 0;
        }

        .footer {
            text-align: center; padding: 16px 0 0;
            color: var(--muted); font-size: 11px;
        }
    </style>
</head>
<body>
    <div class="app">
        <header class="topbar">
            <button id="sidebar-toggle" class="sidebar-toggle" title="Toggle sidebar">&#9776;</button>
            <h1>Subcarrier Power Waterfall</h1>
            <p>antenna 0 &middot; 273 PRBs &times; 12 SCs &times; 14 OFDM symbols</p>
            <div class="spacer"></div>
            <span class="status-pill">
                <span class="status-indicator" id="status-dot"></span>
                <span id="status-text">Connecting&hellip;</span>
            </span>
            <img class="logo" src="/insi-logo" alt="INSI / Northeastern">
        </header>

        <div class="app__main">
            <aside class="sidebar" id="sidebar">

                <details class="sm-group" open>
                    <summary>Spectrum SM <span class="sm-sub">gNB PRB block (RF=1)</span></summary>
                    <div class="sm-body">
                        <div class="sm-row">
                            <label><b>PRB block</b></label>
                            <select id="blk-enabled">
                                <option value="false" selected>disabled</option>
                                <option value="true">enabled</option>
                            </select>
                            <span class="sm-help">forward detector's occupied PRBs to the gNB (UL+DL)</span>
                        </div>
                        <div class="sm-row">
                            <span class="sm-help" id="blk-help">disabled</span>
                        </div>
                    </div>
                </details>

                <details class="sm-group" open>
                    <summary>Adaptive Detector <span class="sm-sub">dApp-local</span></summary>
                    <div class="sm-body">
                        <div class="sm-row">
                            <label><b>enabled</b></label>
                            <select id="det-enabled">
                                <option value="false">false</option>
                                <option value="true" selected>true</option>
                            </select>
                            <span class="sm-help">turn detection on/off</span>
                        </div>
                        <div class="sm-row">
                            <label><b>granularity</b></label>
                            <select id="det-granularity">
                                <option value="prb">PRB (273)</option>
                                <option value="sc" selected>subcarrier (3276)</option>
                            </select>
                            <span class="sm-help">resets history on switch</span>
                        </div>
                        <div class="sm-row">
                            <label><b>SNR threshold</b></label>
                            <input type="number" id="det-threshold-db" value="30" min="0" max="120" step="1">
                            <span class="sm-help">dB above median noise floor</span>
                        </div>
                        <div class="sm-row">
                            <label><b>history depth</b></label>
                            <input type="number" id="det-hist-depth" value="32" min="1" max="512" step="1">
                            <span class="sm-help">frames in median window</span>
                        </div>
                        <div class="sm-row">
                            <label><b>embargo</b></label>
                            <input type="number" id="det-embargo-secs" value="9.9" min="0" max="3600" step="0.1">
                            <span class="sm-help">seconds (post-detect hold)</span>
                        </div>
                        <div class="sm-row">
                            <label><b>L1&times;L2 filter</b></label>
                            <select id="det-mask-sensing">
                                <option value="false">off</option>
                                <option value="true" selected>on</option>
                            </select>
                            <span class="sm-help">mask L1 input to L2 sensing windows</span>
                        </div>
                        <div class="sm-row">
                            <span class="sm-help" id="det-help">disabled</span>
                        </div>
                    </div>
                </details>

                <details class="sm-group" open>
                    <summary>Visualizer <span class="sm-sub">client-side</span></summary>
                    <div class="sm-body">
                        <div class="sm-row">
                            <label><b>freq granularity</b></label>
                            <select id="viz-freq-gran">
                                <option value="sc" selected>subcarrier (3276)</option>
                                <option value="prb">PRB (273)</option>
                            </select>
                            <span class="sm-help">PRB averages 12 SCs</span>
                        </div>
                        <div class="sm-row">
                            <label><b>time granularity</b></label>
                            <select id="viz-time-gran">
                                <option value="sym" selected>symbol (14/slot)</option>
                                <option value="slot">slot</option>
                            </select>
                            <span class="sm-help">slot averages 14 OFDM symbols</span>
                        </div>
                        <div class="sm-row">
                            <label><b>slot subsample</b></label>
                            <input type="number" id="viz-subsample" value="1" min="1" max="10000" step="1">
                            <span class="sm-help">1 = keep all, N = keep 1 in N</span>
                        </div>
                        <div class="sm-row">
                            <label><b>UL slots</b></label>
                            <input type="text" id="viz-slots" value="" placeholder="all" style="width:120px">
                            <span class="sm-help">e.g. 8,9 or 7-12 (empty = all)</span>
                        </div>
                        <div class="sm-row">
                            <label><b>symbols</b></label>
                            <input type="text" id="viz-symbols" value="" placeholder="all" style="width:120px">
                            <span class="sm-help">0-13, e.g. 2-12 (empty = all; n/a in slot mode)</span>
                        </div>
                        <div class="sm-row">
                            <label><b>freq zoom</b></label>
                            <input type="number" id="viz-zoom-min" value="0" min="0" max="272" step="1" style="width:56px">
                            <input type="number" id="viz-zoom-max" value="272" min="0" max="272" step="1" style="width:56px">
                            <span class="sm-help">PRB range min..max (display only)</span>
                        </div>
                        <div class="sm-row">
                            <label><b>display</b></label>
                            <button id="viz-pause-btn" class="rec-btn">&#10073;&#10073; Pause</button>
                            <span class="sm-help" id="viz-pause-help">live</span>
                        </div>
                    </div>
                </details>

                <details class="sm-group">
                    <summary>SigMF Recording <span class="sm-sub">dApp-local</span></summary>
                    <div class="sm-body">
                        <div class="sm-row">
                            <label><b>duration</b></label>
                            <input type="number" id="rec-duration" value="10" min="1" max="86400" step="1">
                            <span class="sm-help">sec</span>
                        </div>
                        <div class="sm-row sm-row-notes">
                            <label><b>notes</b></label>
                            <textarea id="rec-notes" rows="3"
                                placeholder="free text -- embedded as dapp:user_notes in the .sigmf-meta"></textarea>
                        </div>
                        <div class="sm-actions">
                            <button id="rec-btn" class="rec-btn">&#9679; Record</button>
                            <span id="rec-info" class="sm-info"></span>
                        </div>
                    </div>
                </details>

            </aside>

            <section class="dashboard">
                <div class="plot-box density-box">
                    <h2>Spectrum Density</h2>
                    <div class="spec-wrap density">
                        <div class="spec-yaxis">
                            <span>&minus;10 dBFS</span>
                            <span>&minus;60</span>
                            <span>&minus;110 dBFS</span>
                        </div>
                        <canvas id="sc-density" class="spec-canvas"></canvas>
                        <div class="spec-colorbar">
                            <span class="cb-tick">Frequent</span>
                            <div class="cb-gradient"></div>
                            <span class="cb-tick">Rare</span>
                        </div>
                    </div>
                </div>

                <div class="plot-box waterfall-box">
                    <h2>Waterfall</h2>
                    <div class="spec-wrap mask-strip">
                        <div class="mask-label">Blocked</div>
                        <canvas id="mask-canvas"></canvas>
                        <div class="mask-spacer"></div>
                    </div>
                    <div class="spec-wrap waterfall">
                        <div class="spec-yaxis time">
                            <span class="v-text">Newer&nbsp;&uarr;&nbsp;&nbsp;Time</span>
                        </div>
                        <canvas id="sc-waterfall" class="spec-canvas"></canvas>
                        <div class="spec-colorbar">
                            <span class="cb-tick">&minus;10 dBFS</span>
                            <div class="cb-gradient"></div>
                            <span class="cb-tick">&minus;110 dBFS</span>
                        </div>
                    </div>
                    <div class="spec-xaxis" id="wf-xaxis">Subcarrier Index 0 &rarr; 3275 (273 PRBs &times; 12 SCs, antenna 0)</div>
                </div>
            </section>
        </div>

        <footer class="stats-footer">
            <div class="stat-card"><h3>SFN / Slot</h3><p id="stat-sfn-slot">&ndash; / &ndash;</p></div>
            <div class="stat-card"><h3>Frame Rate</h3><p id="stat-rate">&ndash;<span class="unit">Hz</span></p></div>
            <div class="stat-card"><h3>Max Power</h3><p id="stat-max">&ndash;<span class="unit">dBFS</span></p></div>
            <div class="stat-card"><h3>Mean Power</h3><p id="stat-mean">&ndash;<span class="unit">dBFS</span></p></div>
            <div class="stat-card"><h3>Blocked</h3><p id="stat-blocked">&ndash;</p></div>
            <div class="stat-card"><h3>Compute</h3><p id="stat-compute">&ndash;<span class="unit">&micro;s</span></p></div>
            <div class="copyright">&copy; 2026 Northeastern University. All rights reserved.</div>
        </footer>
    </div>

<script>
// Shared data hub: the waterfall IIFE owns the WebSocket and re-emits each
// parsed frame (raw Float32 power values) so the fosphor density panel can
// consume the same data without opening a second connection.
const dataHub = {
    subscribers: [],
    subscribe(fn) { this.subscribers.push(fn); },
    emit(frame) { for (const fn of this.subscribers) fn(frame); },
};

// Shared visualizer view state (client-side only; never sent to the dApp).
// xMin/xMax = normalized [0,1] subcarrier sub-range to display (freq zoom);
// paused freezes the waterfall + density; slots/symbols are display filters
// (a Set of selected indices, or null = all).
const vizState = { xMin: 0.0, xMax: 1.0, paused: false, slots: null, symbols: null };
// true if v is selected (null / empty set = "all").
function vizInSet(set, v) { return !set || set.size === 0 || set.has(v); }

// Browser-side per-stage timing — each stage adds its duration; the /clientstats
// reporter sends these ~1/s then resets. Summed over the ~1s window, each *_Ms ≈
// ms/s of CPU for that stage. worker runs on its own thread; main + dens + wf
// share the main thread (sum near 1000 ms/s = main thread saturated).
const perf = { workerMs: 0, workerN: 0, mainMs: 0, mainN: 0,
               densMs: 0, densN: 0, wfMs: 0, wfN: 0 };

// dB conversion is now done in the dApp (sends u8 dBFS quantized over [db_min, db_max]
// reported in the WS header). The visualizer only needs to map u8 → NDC / display.

(function() {
    // -------- Config --------
    // N_ANTS is fixed (the dApp only publishes antenna 0). N_SYMBOLS / N_SC are
    // the maximum (full-resolution) values used to size WebGL resources; the
    // actual values per indication come from the WS header and may be smaller
    // when the dApp publishes aggregated PRB / slot data.
    const N_ANTS    = 1;     // dApp now publishes only antenna 0
    const N_SYMBOLS = 14;
    const N_SC      = __N_SC__;  // server-injected: num_prbs*12 for the active bandwidth
    const N_ROWS    = 512;
    const N_STREAMS = __N_STREAMS__;  // parallel WS connections (frame-interleave); server-injected
    const COMPRESS  = (__COMPRESS__ === 1);  // server-injected: WS batches are deflate-compressed (VIZ_COMPRESS)
    const SID = Math.random().toString(36).slice(2) + Date.now().toString(36);  // links this page's WS streams to its /clientstats (end-to-end pacing)
    // dBFS window: 0 dBFS = full scale (FP16 max), -110 dBFS = noise floor / below.
    const DB_MIN = -110, DB_MAX = -10;
    const STALE_MS = 1500;

    const canvas = document.getElementById('sc-waterfall');
    const gl = canvas.getContext('webgl2', { antialias: false, premultipliedAlpha: false });
    if (!gl) {
        canvas.outerHTML = '<div style="color:#dc3545;padding:20px;">WebGL2 not supported in this browser.</div>';
        return;
    }

    const statusDot  = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    function setStatus(cls, txt) { statusDot.className = 'status-indicator ' + cls; statusText.textContent = txt; }
    setStatus('status-waiting', 'Connecting…');

    // -------- Shaders --------
    const VS_SRC = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
    v_uv = a_pos * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

    const FS_SRC = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_data;
uniform float u_writeRow;
uniform float u_subRow;
uniform float u_nRows;
uniform float u_filledRows;
uniform float u_xMin;   // freq-zoom: normalized [0,1] subcarrier sub-range to display
uniform float u_xMax;

// Turbo colormap (Google Research) — fosphor-style thermal:
// black/navy -> blue -> cyan -> green -> yellow -> orange -> red.
vec3 turbo(float x) {
    x = clamp(x, 0.0, 1.0);
    const vec4 kR4 = vec4( 0.13572138,   4.61539260, -42.66032258, 132.13108234);
    const vec4 kG4 = vec4( 0.09140261,   2.19418839,   4.84296658, -14.18503333);
    const vec4 kB4 = vec4( 0.10667330,  12.64194608, -60.58204836, 110.36276771);
    const vec2 kR2 = vec2(-152.94239396, 59.28637943);
    const vec2 kG2 = vec2(   4.27729857,  2.82956604);
    const vec2 kB2 = vec2( -89.90310912, 27.34824973);
    vec4 v4 = vec4(1.0, x, x*x, x*x*x);
    vec2 v2 = v4.zw * v4.z;  // x^4, x^5
    return vec3(
        dot(v4, kR4) + dot(v2, kR2),
        dot(v4, kG4) + dot(v2, kG2),
        dot(v4, kB4) + dot(v2, kB2)
    );
}

void main() {
    // Single waterfall (mean across antennas). Top = newest, scroll continuous at arrival snap.
    float depthFromTop = (1.0 - v_uv.y) * (u_nRows - 1.0);
    float dataRow = u_writeRow - 2.0 + u_subRow - depthFromTop;
    float age = (u_writeRow - 1.0) - dataRow;
    if (age >= u_filledRows) {
        outColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    float row = mod(dataRow + u_nRows * 1024.0, u_nRows);
    float xz = u_xMin + v_uv.x * (u_xMax - u_xMin);   // freq zoom
    vec2 uv = vec2(xz, (row + 0.5) / u_nRows);
    float t = texture(u_data, uv).r;
    outColor = vec4(turbo(t), 1.0);
}`;

    function compile(type, src) {
        const sh = gl.createShader(type);
        gl.shaderSource(sh, src);
        gl.compileShader(sh);
        if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
            console.error('Shader error:', gl.getShaderInfoLog(sh));
        }
        return sh;
    }
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS_SRC));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS_SRC));
    gl.bindAttribLocation(prog, 0, 'a_pos');
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        console.error('Link error:', gl.getProgramInfoLog(prog));
    }
    gl.useProgram(prog);

    const u_data       = gl.getUniformLocation(prog, 'u_data');
    const u_writeRow   = gl.getUniformLocation(prog, 'u_writeRow');
    const u_subRow     = gl.getUniformLocation(prog, 'u_subRow');
    const u_nRows      = gl.getUniformLocation(prog, 'u_nRows');
    const u_filledRows = gl.getUniformLocation(prog, 'u_filledRows');
    const u_xMin       = gl.getUniformLocation(prog, 'u_xMin');
    const u_xMax       = gl.getUniformLocation(prog, 'u_xMax');
    gl.uniform1i(u_data, 0);
    gl.uniform1f(u_nRows, N_ROWS);
    gl.uniform1f(u_xMin, 0.0);
    gl.uniform1f(u_xMax, 1.0);

    // Fullscreen quad
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
        -1, -1,  1, -1, -1,  1,
        -1,  1,  1, -1,  1,  1,
    ]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    // 2D texture: width=N_SC, height=N_ROWS, R8. One row per slot, antennas pre-aggregated.
    const tex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R8, N_SC, N_ROWS);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);

    const dbSpan = DB_MAX - DB_MIN;

    // -------- State --------
    let writeRow = 0;
    let filledRows = 0;
    let lastArrival = 0;
    let interArrivalMs = 100;
    let lastDataMs = 0;

    // FPS / Hz tracking
    let recvCount = 0;
    let lastRateUpdate = performance.now();
    let smoothedRate = 0;

    // Rows staged for the next rAF are written directly into one reusable
    // buffer (no per-row allocation — critical at ~11k rows/s in per-symbol
    // mode). pendingCount rows are uploaded to the waterfall texture in a single
    // shot per frame (perf #2). Capacity is N_ROWS (the texture height); older
    // rows could never be displayed anyway.
    const pendingBuf = new Uint8Array(N_ROWS * N_SC);
    let pendingCount = 0;
    let lastSfn = 0, lastSlot = 0, lastMaxDb = 0, lastMeanDb = 0;

    // Expand one source row (nSc values, each repeated `factor` times for PRB
    // mode) into the fixed-width N_SC texture row at dst[dstOff..dstOff+N_SC).
    function expandRowInto(dst, dstOff, src, srcOff, nSc, factor) {
        if (factor === 1) {
            dst.set(src.subarray(srcOff, srcOff + nSc), dstOff);
            return;
        }
        let o = dstOff;
        for (let p = 0; p < nSc; p++) {
            const v = src[srcOff + p];
            for (let k = 0; k < factor; k++) dst[o++] = v;
        }
    }

    function resize() {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
        const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w; canvas.height = h;
            gl.viewport(0, 0, w, h);
        }
    }
    window.addEventListener('resize', resize);

    // -------- WebSocket receiver (perf #3: in a Web Worker, with backpressure) --------
    // The WS connection + batch parse + waterfall-row assembly run OFF the main
    // thread. The worker posts (≈60/s) a coalesced row-block (transferable) for
    // the waterfall + the LATEST frame for the dataHub fan-out. An ack-gated cap
    // (MAX_OUT) bounds how far ahead the worker can post, so when the main thread
    // can't keep up the worker DROPS whole batches (counted) instead of piling up
    // buffers — overload stays bounded and recovers on its own. expandRowInto
    // (above) is unused on the main thread now; the worker does its own expansion.
    const wsUrl = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
    setStatus('status-waiting', 'Connecting…');

    const workerSrc = `
let N_SC=0,N_ROWS=0,N_ANTS=0,N_SYMBOLS=0,N_STREAMS=1,wsUrl='';
let slots=null,symbols=null,paused=false;
let rowBuf=null,rowCount=0;
let latestRaw=null,latestDet=null,latestMeta=null;   // fan-out: only the LATEST frame per post
let lastSfn=0,lastSlot=0,lastMax=0,lastMean=0;
let outstanding=0; let MAX_OUT=2;   // BACKPRESSURE: cap unacked posts -> bounds the main-thread heap (scaled to N_STREAMS)
let dropped=0, consumed=0;            // cumulative frames dropped (main behind) / consumed
let procT0=0;                         // perf: start of this batch's processing
let COMPRESS=false;                   // server-injected: WS batches are deflate-compressed
let SID='';                           // page id added to each WS query for end-to-end pacing
let procChain=Promise.resolve();      // serialize async inflate+parse across ALL streams (no rowBuf race)
// Inflate a deflate-compressed batch with the native DecompressionStream.
function inflate(ab){
  return new Response(new Blob([ab]).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer();
}
function inSet(a,v){return !a||a.length===0||a.indexOf(v)>=0;}
function connectStream(k){
  // Frame-interleave: stream k of N_STREAMS receives only frames idx%N_STREAMS==k.
  // Each socket is its own TCP connection => its own send window; the worker
  // merges the whole frames from all streams in arrival order.
  const url=wsUrl+'?stream='+k+'&of='+N_STREAMS+(SID?('&sid='+encodeURIComponent(SID)):'');
  const ws=new WebSocket(url); ws.binaryType='arraybuffer';
  // Connect watchdog: a handshake left HANGING (route/proxy holding half-open
  // sockets after a visualizer restart) never fires onclose, so the 1s retry
  // below would wait forever — the "dashboard takes ~40s to connect" case.
  // Abort a socket still CONNECTING after 3s; close() fires onclose -> retry.
  setTimeout(function(){ if(ws.readyState===WebSocket.CONNECTING) ws.close(); },3000);
  ws.onclose=function(){setTimeout(function(){connectStream(k);},1000);};
  ws.onmessage=function(ev){
    if(paused) return;
    if(COMPRESS){ const cb=ev.data; procChain=procChain.then(function(){return inflate(cb);}).then(parseMsg).catch(function(){}); }
    else { parseMsg(ev.data); }
  };
}
// Parse one (decompressed) batch + accumulate rows. Synchronous; when COMPRESS
// it runs via procChain so streams never touch the shared rowBuf concurrently.
function parseMsg(buf){
    if(!(buf instanceof ArrayBuffer)||buf.byteLength<2) return;
    const dv=new DataView(buf);
    const nFrames=dv.getUint16(0,true);
    // Backpressure: if the main thread hasn't ack'd the last MAX_OUT posts it
    // can't keep up -> DROP this whole batch (don't parse/accumulate) so posts
    // and heap stay bounded. Graceful + self-recovering: posting resumes the
    // moment an ack frees a slot. This is what makes overload NOT sticky.
    if(outstanding>=MAX_OUT){ dropped+=nFrames; return; }
    procT0=performance.now();
    let off=2;
    for(let fi=0;fi<nFrames;fi++){
      if(off+14>buf.byteLength) break;
      const sfn=dv.getUint16(off,true),slot=dv.getUint16(off+2,true);
      const nAnt=dv.getUint16(off+4,true),nSym=dv.getUint16(off+6,true),nSc=dv.getUint16(off+8,true);
      const dbMin=dv.getInt8(off+10),dbMax=dv.getInt8(off+11),kernelUs=dv.getUint16(off+12,true);
      const powerLen=nAnt*nSym*nSc,powerOff=off+14;
      if(powerOff+powerLen+1>buf.byteLength) break;
      const detTag=dv.getUint8(powerOff+powerLen); let detN=0,trailerLen=1;
      if(detTag===1){ if(powerOff+powerLen+3>buf.byteLength) break; detN=dv.getUint16(powerOff+powerLen+1,true); trailerLen=3+detN; if(powerOff+powerLen+trailerLen>buf.byteLength) break; }
      off=powerOff+powerLen+trailerLen;
      if(nAnt!==N_ANTS) continue;
      if(nSym<1||nSym>N_SYMBOLS) continue;  // accept mixed-slot frames (e.g. 4 sym), not just 1 or full-slot
      if(nSc===0||(N_SC%nSc)!==0||nSc>N_SC) continue;
      if(detTag!==0&&detTag!==1) continue;
      if(detTag===1&&detN!==N_SC&&detN!==(N_SC/12)) continue;
      if(!inSet(slots,slot)) continue;
      const data=new Uint8Array(buf,powerOff,powerLen);
      // Keep only the latest frame for the panel fan-out (density shows latest;
      // stats are throttled) — far less transfer + main-thread work than all 13.
      latestRaw=data.slice();
      latestDet=detTag===1? new Uint8Array(buf,powerOff+powerLen+3,detN).slice():null;
      latestMeta={sfn:sfn,slot:slot,nAnt:nAnt,nSym:nSym,nSc:nSc,dbMin:dbMin,dbMax:dbMax,kernelUs:kernelUs,detN:detN};
      const factor=(N_SC/nSc)|0; let maxU8=0,sumU8=0;
      for(let i=0;i<powerLen;i++){const v=data[i]; if(v>maxU8)maxU8=v; sumU8+=v;}
      for(let s=0;s<nSym;s++){
        if(nSym>1&&!inSet(symbols,s)) continue;
        if(rowCount>=N_ROWS) break;
        const dstOff=rowCount*N_SC,srcOff=s*nSc;
        if(factor===1){ rowBuf.set(data.subarray(srcOff,srcOff+nSc),dstOff); }
        else { let o=dstOff; for(let p=0;p<nSc;p++){const v=data[srcOff+p]; for(let k=0;k<factor;k++) rowBuf[o++]=v;} }
        rowCount++;
      }
      const span=dbMax-dbMin; lastSfn=sfn; lastSlot=slot; lastMax=(maxU8/255)*span+dbMin; lastMean=(sumU8/(powerLen*255))*span+dbMin;
    }
    consumed+=nFrames;
    flush();
}
function flush(){
  if(rowCount===0&&!latestRaw) return;
  const rows=rowBuf.slice(0,rowCount*N_SC);
  const raw = latestRaw ? latestRaw : new Uint8Array(0);
  const det = latestDet || null;
  const transfer = det ? [rows.buffer, raw.buffer, det.buffer] : [rows.buffer, raw.buffer];
  postMessage({type:'block',rows:rows.buffer,rowCount:rowCount,lastSfn:lastSfn,lastSlot:lastSlot,lastMax:lastMax,lastMean:lastMean,
               raw:raw.buffer, det: det?det.buffer:null, meta:latestMeta, dropped:dropped, consumed:consumed,
               procMs: performance.now()-procT0}, transfer);
  outstanding++;
  rowCount=0; latestRaw=null; latestDet=null; latestMeta=null;
}
onmessage=function(e){
  const m=e.data;
  if(m.type==='init'){ N_SC=m.N_SC;N_ROWS=m.N_ROWS;N_ANTS=m.N_ANTS;N_SYMBOLS=m.N_SYMBOLS;N_STREAMS=m.N_STREAMS||1;wsUrl=m.wsUrl;COMPRESS=!!m.COMPRESS;SID=m.SID||'';
    MAX_OUT=Math.max(2,N_STREAMS+1);   // allow ~1 in-flight post per stream (main thread is far from saturated)
    rowBuf=new Uint8Array(N_ROWS*N_SC); for(let k=0;k<N_STREAMS;k++) connectStream(k); }
  else if(m.type==='filters'){ slots=m.slots; symbols=m.symbols; paused=m.paused; }
  else if(m.type==='ack'){ if(outstanding>0) outstanding--; }   // main drained a post -> free a slot
};
`;

    const _wkr = new Worker(URL.createObjectURL(new Blob([workerSrc], { type: 'application/javascript' })));
    let g_consumed = 0, g_dropped = 0;   // cumulative; reported to the server for the ceiling measurement
    _wkr.onmessage = (e) => {
        const m = e.data;
        if (!m || m.type !== 'block') return;
        const _t0 = performance.now();
        // Stage the worker's pre-assembled rows into pendingBuf; the rAF does the
        // coalesced texture upload (perf #2).
        const rows = new Uint8Array(m.rows);
        const avail = (rows.length / N_SC) | 0;
        const take = Math.min(avail, N_ROWS - pendingCount);
        if (take > 0) {
            pendingBuf.set(rows.subarray(0, take * N_SC), pendingCount * N_SC);
            pendingCount += take;
        }
        if (m.rowCount > 0) {
            lastSfn = m.lastSfn; lastSlot = m.lastSlot; lastMaxDb = m.lastMax; lastMeanDb = m.lastMean;
        }
        // Fan out to the other panels. The detector/stats use the LATEST frame;
        // the density (Option B) integrates the FULL row-block (rowsU8/rowsCount)
        // that the waterfall also consumes — every frame, no extra transfer.
        if (m.meta) {
            const fm = m.meta;
            const dataU8 = new Uint8Array(m.raw);
            const detMask = m.det ? new Uint8Array(m.det) : null;
            dataHub.emit({ sfn: fm.sfn, slot: fm.slot, nAnt: fm.nAnt, nSym: fm.nSym, nSc: fm.nSc,
                           dbMin: fm.dbMin, dbMax: fm.dbMax, dataU8, detMask, detN: fm.detN, kernelUs: fm.kernelUs,
                           rowsU8: rows, rowsCount: avail });
        }
        g_consumed = m.consumed; g_dropped = m.dropped;
        perf.mainMs += performance.now() - _t0; perf.mainN++;
        if (m.procMs !== undefined) { perf.workerMs += m.procMs; perf.workerN++; }
        _wkr.postMessage({ type: 'ack' });   // backpressure: this post is drained, free a slot
    };
    // Report the browser's true consume/drop counts to the server (~1/s) so the
    // smooth-rate ceiling is measurable even though the WS TCP buffer always drains.
    // Pacing feedback: a light {sid, consumed} beacon every 100 ms so the server
    // can bound in-flight tightly (and not over-shed at high rate); fold in the
    // full perf snapshot once a second.
    let _statTick = 0;
    setInterval(() => {
        _statTick++;
        const body = { sid: SID, consumed: g_consumed, dropped: g_dropped, paused: (vizState.paused || document.hidden) };
        if (_statTick % 10 === 0) {
            body.worker_ms = Math.round(perf.workerMs); body.worker_n = perf.workerN;
            body.main_ms = Math.round(perf.mainMs);     body.main_n = perf.mainN;
            body.dens_ms = Math.round(perf.densMs);     body.dens_n = perf.densN;
            body.wf_ms = Math.round(perf.wfMs);         body.wf_n = perf.wfN;
            perf.workerMs = perf.workerN = perf.mainMs = perf.mainN = 0;
            perf.densMs = perf.densN = perf.wfMs = perf.wfN = 0;
        }
        fetch('/clientstats', { method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body) }).catch(() => {});
    }, 100);
    // Fire an immediate beacon on hide/show so the server learns the paused state
    // even when interval timers are throttled in a background tab -> the feed
    // resumes instantly (re-baselined) on refocus instead of staying frozen.
    document.addEventListener('visibilitychange', () => {
        fetch('/clientstats', { method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sid: SID, consumed: g_consumed, dropped: g_dropped,
                                   paused: (vizState.paused || document.hidden) }) }).catch(() => {});
    });
    _wkr.postMessage({ type: 'init', N_SC, N_ROWS, N_ANTS, N_SYMBOLS, N_STREAMS, COMPRESS, SID, wsUrl });
    // Keep the worker's filters/paused in sync without hooking every UI mutation
    // site; the timer also runs (throttled) when hidden so the worker pauses.
    setInterval(() => {
        _wkr.postMessage({ type: 'filters',
            slots:   vizState.slots   ? Array.from(vizState.slots)   : null,
            symbols: vizState.symbols ? Array.from(vizState.symbols) : null,
            paused:  vizState.paused || document.hidden });
    }, 150);

    // -------- Render loop --------
    function frame(now) {
        const _wfT0 = performance.now();
        resize();

        if (pendingCount > 0) {
            const k = Math.min(pendingCount, N_ROWS);
            gl.bindTexture(gl.TEXTURE_2D, tex);
            // Coalesced upload (perf #2): push all k staged rows starting at
            // writeRow in at most TWO texSubImage2D calls (split at the ring
            // wrap), instead of one call per row.
            const first = Math.min(k, N_ROWS - writeRow);
            gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, writeRow, N_SC, first,
                             gl.RED, gl.UNSIGNED_BYTE, pendingBuf.subarray(0, first * N_SC));
            if (k > first) {
                gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, N_SC, k - first,
                                 gl.RED, gl.UNSIGNED_BYTE, pendingBuf.subarray(first * N_SC, k * N_SC));
            }
            writeRow = (writeRow + k) % N_ROWS;
            filledRows = Math.min(N_ROWS, filledRows + k);
            recvCount += k;

            if (lastArrival > 0) {
                const dt = (now - lastArrival) / k;
                if (dt > 0 && dt < 5000) {
                    interArrivalMs = 0.85 * interArrivalMs + 0.15 * dt;
                }
            }
            lastArrival = now;
            lastDataMs = now;

            // Update stat cards from the latest staged row.
            document.getElementById('stat-sfn-slot').textContent = lastSfn + ' / ' + lastSlot;
            document.getElementById('stat-max').innerHTML  = lastMaxDb.toFixed(1)  + '<span class="unit">dBFS</span>';
            document.getElementById('stat-mean').innerHTML = lastMeanDb.toFixed(1) + '<span class="unit">dBFS</span>';

            pendingCount = 0;
        }

        // Frame rate (smoothed, 1Hz update).
        if (now - lastRateUpdate > 1000) {
            const inst = recvCount * 1000 / (now - lastRateUpdate);
            smoothedRate = 0.6 * smoothedRate + 0.4 * inst;
            recvCount = 0;
            lastRateUpdate = now;
            document.getElementById('stat-rate').innerHTML = smoothedRate.toFixed(1) + '<span class="unit">Hz</span>';
        }

        // Status: active vs stale (WS lives in the worker now, so go by data).
        if (lastDataMs > 0) {
            if ((now - lastDataMs) < STALE_MS) {
                setStatus('status-active', 'Live — receiving data');
            } else {
                setStatus('status-stale', 'Live — waiting for data');
            }
        }

        let subRow = 0;
        if (lastArrival > 0 && interArrivalMs > 0) {
            subRow = (now - lastArrival) / interArrivalMs;
            if (subRow < 0) subRow = 0;
            if (subRow > 1) subRow = 1;
        }

        gl.useProgram(prog);
        gl.uniform1f(u_writeRow, writeRow);
        gl.uniform1f(u_subRow, subRow);
        gl.uniform1f(u_filledRows, filledRows);
        gl.uniform1f(u_xMin, vizState.xMin);
        gl.uniform1f(u_xMax, vizState.xMax);
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        perf.wfMs += performance.now() - _wfT0; perf.wfN++;
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
})();

// --- Spectrum density (fosphor-style persistence) ---
(function() {
    const N_ANTS = 1, N_SYMBOLS = 14, N_SC = __N_SC__;
    const N_BINS = 256;                  // dB resolution of the accumulation FBO
    const DB_MIN = -110, DB_MAX = -10;   // dBFS window matching the waterfall
    const DB_SPAN = DB_MAX - DB_MIN;
    const DECAY = 0.93;                  // per-rAF decay; ~14-frame e-fold
    // Per-symbol single-antenna: ~3.5× more points than the original slot-averaged
    // 4-antenna version, so increment is scaled down accordingly.
    const INC = 0.02;
    const NORM_MAX = INC / (1 - DECAY);

    const canvas = document.getElementById('sc-density');
    const gl = canvas.getContext('webgl2', { antialias: false, premultipliedAlpha: false });
    if (!gl) {
        canvas.outerHTML = '<div style="color:#dc3545;padding:20px;">WebGL2 not supported.</div>';
        return;
    }
    if (!gl.getExtension('EXT_color_buffer_float')) {
        canvas.outerHTML = '<div style="color:#dc3545;padding:20px;">' +
            'EXT_color_buffer_float not supported — fosphor density requires float framebuffer rendering.</div>';
        return;
    }

    // -------- Shaders --------
    const VS_QUAD = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
    v_uv = a_pos * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

    const FS_DECAY = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_src;
uniform float u_decay;
void main() {
    float v = texture(u_src, v_uv).r;
    outColor = vec4(v * u_decay, 0.0, 0.0, 0.0);
}`;

    // Vertex shader for accumulation: gl_VertexID gives the (antenna, subcarrier)
    // index; modulo N_SC pulls out the subcarrier — antennas overlay each other.
    const VS_ACC = `#version 300 es
in float a_y;
uniform float u_n_sc;
void main() {
    int idx = gl_VertexID;
    int sc = idx - (idx / int(u_n_sc)) * int(u_n_sc);  // % using int math
    float x = (float(sc) + 0.5) / u_n_sc * 2.0 - 1.0;
    gl_Position = vec4(x, a_y, 0.0, 1.0);
    gl_PointSize = 2.0;
}`;

    const FS_ACC = `#version 300 es
precision highp float;
out vec4 outColor;
uniform float u_inc;
void main() {
    outColor = vec4(u_inc, 0.0, 0.0, 0.0);
}`;

    const FS_DISP = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_src;
uniform float u_norm;
uniform float u_xMin;   // freq-zoom: normalized [0,1] subcarrier sub-range to display
uniform float u_xMax;

// Turbo colormap (Google Research) — fosphor-style thermal:
// black/navy -> blue -> cyan -> green -> yellow -> orange -> red.
vec3 turbo(float x) {
    x = clamp(x, 0.0, 1.0);
    const vec4 kR4 = vec4( 0.13572138,   4.61539260, -42.66032258, 132.13108234);
    const vec4 kG4 = vec4( 0.09140261,   2.19418839,   4.84296658, -14.18503333);
    const vec4 kB4 = vec4( 0.10667330,  12.64194608, -60.58204836, 110.36276771);
    const vec2 kR2 = vec2(-152.94239396, 59.28637943);
    const vec2 kG2 = vec2(   4.27729857,  2.82956604);
    const vec2 kB2 = vec2( -89.90310912, 27.34824973);
    vec4 v4 = vec4(1.0, x, x*x, x*x*x);
    vec2 v2 = v4.zw * v4.z;  // x^4, x^5
    return vec3(
        dot(v4, kR4) + dot(v2, kR2),
        dot(v4, kG4) + dot(v2, kG2),
        dot(v4, kB4) + dot(v2, kB2)
    );
}

void main() {
    vec2 uv = vec2(u_xMin + v_uv.x * (u_xMax - u_xMin), v_uv.y);   // freq zoom
    float v = texture(u_src, uv).r / u_norm;
    v = clamp(v, 0.0, 1.0);
    if (v < 0.003) {
        outColor = vec4(0.0, 0.0, 0.0, 1.0);  // black background — fosphor look
        return;
    }
    outColor = vec4(turbo(v), 1.0);
}`;

    function compile(type, src) {
        const sh = gl.createShader(type);
        gl.shaderSource(sh, src);
        gl.compileShader(sh);
        if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
            console.error('density shader error:', gl.getShaderInfoLog(sh));
        }
        return sh;
    }
    function makeProg(vs, fs, attrName) {
        const p = gl.createProgram();
        gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
        gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
        gl.bindAttribLocation(p, 0, attrName);
        gl.linkProgram(p);
        if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
            console.error('density link error:', gl.getProgramInfoLog(p));
        }
        return p;
    }

    const progDecay = makeProg(VS_QUAD, FS_DECAY, 'a_pos');
    const progAcc   = makeProg(VS_ACC,  FS_ACC,  'a_y');
    const progDisp  = makeProg(VS_QUAD, FS_DISP, 'a_pos');

    const u_decay_src   = gl.getUniformLocation(progDecay, 'u_src');
    const u_decay_decay = gl.getUniformLocation(progDecay, 'u_decay');
    const u_acc_n_sc    = gl.getUniformLocation(progAcc,   'u_n_sc');
    const u_acc_inc     = gl.getUniformLocation(progAcc,   'u_inc');
    const u_disp_src    = gl.getUniformLocation(progDisp,  'u_src');
    const u_disp_norm   = gl.getUniformLocation(progDisp,  'u_norm');
    const u_disp_xMin   = gl.getUniformLocation(progDisp,  'u_xMin');
    const u_disp_xMax   = gl.getUniformLocation(progDisp,  'u_xMax');

    // Fullscreen-quad VBO for decay + display.
    const quadVbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadVbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
        -1, -1,  1, -1, -1,  1,
        -1,  1,  1, -1,  1,  1,
    ]), gl.STATIC_DRAW);

    // Per-point Y VBO (refilled when new data arrives). N_ANTS × N_SYMBOLS × N_SC
    // points per draw — antennas and symbols overlay because the vertex shader
    // pulls subcarrier index via `gl_VertexID % N_SC`, and sc varies fastest in
    // the data layout [ant][sym][sc].
    const yVbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, yVbo);
    gl.bufferData(gl.ARRAY_BUFFER, N_ANTS * N_SYMBOLS * N_SC * 4, gl.DYNAMIC_DRAW);

    // Two ping-pong R16F framebuffers for the accumulation buffer.
    function makeAccumTex() {
        const t = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R16F, N_SC, N_BINS);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        return t;
    }
    function makeFbo(tex) {
        const fb = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
        return fb;
    }
    let texA = makeAccumTex(), texB = makeAccumTex();
    let fboA = makeFbo(texA),  fboB = makeFbo(texB);
    // Clear both — texStorage2D leaves contents undefined.
    for (const fb of [fboA, fboB]) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
        gl.viewport(0, 0, N_SC, N_BINS);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
    }
    let src = { tex: texA, fbo: fboA };
    let dst = { tex: texB, fbo: fboB };

    let lastFrameMs = performance.now();

    function resize() {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
        const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w; canvas.height = h;
        }
    }
    window.addEventListener('resize', resize);

    // Option B: integrate EVERY frame. Accumulate the full row-block the worker
    // assembled this post (the SAME expanded rows the waterfall gets) into a
    // pending float buffer; render() draws them all in one accumulation pass.
    // Brightness is normalized by frames-per-refresh (normEMA) so the density
    // doesn't blow out as the rate rises. Rows are u8 in [DB_MIN,DB_MAX] →
    // (u8/127.5)-1 maps straight to the display NDC.
    const MAX_DENSE_ROWS = 512;   // cap on rows accumulated per refresh (density-local; N_ROWS lives in the waterfall IIFE)
    const densePending = new Float32Array(MAX_DENSE_ROWS * N_SC);
    let denseRows = 0;
    let normEMA = 1.0;
    dataHub.subscribe(frame => {
        const rows = frame.rowsU8;
        let rc = frame.rowsCount || 0;
        if (!rows || rc <= 0) return;
        if (denseRows + rc > MAX_DENSE_ROWS) rc = MAX_DENSE_ROWS - denseRows;   // cap (backpressure keeps this rare)
        if (rc <= 0) return;
        const base = denseRows * N_SC, n = rc * N_SC;
        for (let i = 0; i < n; i++) densePending[base + i] = (rows[i] / 127.5) - 1.0;
        denseRows += rc;
    });

    function render(now) {
        const _dT0 = performance.now();
        resize();

        // 1) Decay pass: src → dst (multiplied by DECAY).
        gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fbo);
        gl.viewport(0, 0, N_SC, N_BINS);
        gl.disable(gl.BLEND);
        gl.useProgram(progDecay);
        gl.bindBuffer(gl.ARRAY_BUFFER, quadVbo);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, src.tex);
        gl.uniform1i(u_decay_src, 0);
        gl.uniform1f(u_decay_decay, DECAY);
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        // 2) Accumulation pass — draw EVERY row accumulated since the last
        // refresh (Option B), spread across the full width (rows are N_SC-wide).
        if (denseRows > 0) {
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.ONE, gl.ONE);
            gl.useProgram(progAcc);
            gl.uniform1f(u_acc_n_sc, N_SC);
            gl.uniform1f(u_acc_inc, INC);
            gl.bindBuffer(gl.ARRAY_BUFFER, yVbo);
            gl.bufferData(gl.ARRAY_BUFFER, densePending.subarray(0, denseRows * N_SC), gl.DYNAMIC_DRAW);
            gl.enableVertexAttribArray(0);
            gl.vertexAttribPointer(0, 1, gl.FLOAT, false, 0, 0);
            gl.drawArrays(gl.POINTS, 0, denseRows * N_SC);
            gl.disable(gl.BLEND);
            // Normalize brightness by frames-per-refresh (denseRows/N_SYMBOLS),
            // smoothed, so integrating ~Nx more samples doesn't blow out the plot.
            normEMA = 0.9 * normEMA + 0.1 * Math.max(1, denseRows / N_SYMBOLS);
            denseRows = 0;
        }

        // 3) Swap so next frame's decay reads what we just wrote.
        const tmp = src; src = dst; dst = tmp;

        // 4) Display pass: render src to canvas with colormap.
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.useProgram(progDisp);
        gl.bindBuffer(gl.ARRAY_BUFFER, quadVbo);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, src.tex);
        gl.uniform1i(u_disp_src, 0);
        gl.uniform1f(u_disp_norm, NORM_MAX * normEMA);
        gl.uniform1f(u_disp_xMin, vizState.xMin);
        gl.uniform1f(u_disp_xMax, vizState.xMax);
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        lastFrameMs = now;
        perf.densMs += performance.now() - _dT0; perf.densN++;
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
})();

// --- SigMF recording controls ---
(function() {
    const btn  = document.getElementById('rec-btn');
    const dur  = document.getElementById('rec-duration');
    const info = document.getElementById('rec-info');
    let pollT = null;
    let recording = false;

    function fmtActive(s) {
        const el = (s.elapsed_s ?? 0).toFixed(1);
        const dr = (s.duration_s ?? 0).toFixed(0);
        const mb = ((s.bytes ?? 0) / 1e6).toFixed(1);
        return el + '/' + dr + 's · ' + (s.slots ?? 0) + ' slots · ' + mb + ' MB · ' + (s.filename ?? '');
    }
    function setActive(s) {
        recording = true;
        btn.textContent = '■ Stop';
        btn.classList.add('rec-active');
        dur.disabled = true;
        info.textContent = fmtActive(s);
    }
    function setIdle(s) {
        recording = false;
        btn.textContent = '● Record';
        btn.classList.remove('rec-active');
        dur.disabled = false;
        if (s && s.error) {
            info.textContent = 'Error: ' + s.error;
        } else if (s && s.filename) {
            const mb = ((s.bytes ?? 0) / 1e6).toFixed(1);
            info.textContent = 'Saved ' + s.filename + ' (' + (s.slots ?? '?') + ' slots, ' + mb + ' MB)' +
                (s.reason ? ' — ' + s.reason : '');
        } else {
            info.textContent = '';
        }
        if (pollT) { clearInterval(pollT); pollT = null; }
    }

    async function poll() {
        try {
            const r = await fetch('/record/status').then(x => x.json());
            if (r.state === 'recording') {
                setActive(r);
            } else if (recording) {
                // dApp auto-stopped (duration limit reached server-side)
                setIdle({ filename: r.filename, slots: r.slots, bytes: r.bytes, reason: 'duration limit reached' });
            }
        } catch (e) { /* transient */ }
    }

    btn.addEventListener('click', async () => {
        btn.disabled = true;
        try {
            if (recording) {
                const r = await fetch('/record/stop', { method: 'POST' }).then(x => x.json());
                setIdle(r.ok ? r : { error: r.error || 'stop failed' });
            } else {
                const ds = parseFloat(dur.value);
                if (!(ds > 0)) { alert('Duration must be > 0 seconds'); return; }
                const notesEl = document.getElementById('rec-notes');
                const notes = notesEl ? notesEl.value : '';
                const r = await fetch('/record/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ duration_s: ds, notes: notes }),
                }).then(x => x.json());
                if (r.ok) {
                    setActive({ elapsed_s: 0, duration_s: ds, filename: r.filename, slots: 0, bytes: 0 });
                    pollT = setInterval(poll, 1000);
                } else {
                    info.textContent = 'Error: ' + (r.error || 'start failed');
                }
            }
        } finally {
            btn.disabled = false;
        }
    });

    // Reflect existing state on page load (e.g. if visualizer reconnected mid-recording).
    fetch('/record/status').then(r => r.json()).then(r => {
        if (r && r.state === 'recording') {
            setActive(r);
            pollT = setInterval(poll, 1000);
        }
    }).catch(() => {});
})();

// --- Visualizer publish controls (granularity + subsample) ---
// Auto-applied on change. Posts to /viz/apply which forwards to the dApp
// over the existing REQ/REP control socket. The dApp owns the actual
// aggregation + frame-dropping, so the wire payload shrinks accordingly.
(function() {
    const freqEl = document.getElementById('viz-freq-gran');
    const timeEl = document.getElementById('viz-time-gran');
    const subEl  = document.getElementById('viz-subsample');
    if (!freqEl || !timeEl || !subEl) return;

    async function apply() {
        const n = parseInt(subEl.value, 10);
        const subN = Number.isFinite(n) && n >= 1 ? n : 1;
        const cfg = {
            freq_gran: freqEl.value,
            time_gran: timeEl.value,
            subsample_n: subN,
        };
        try {
            await fetch('/viz/apply', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config: cfg }),
            });
        } catch (_) { /* transient — next change will retry */ }
    }
    freqEl.addEventListener('change', apply);
    timeEl.addEventListener('change', apply);
    subEl.addEventListener('change', apply);
    // Push the initial state once on load so the dApp matches the UI defaults
    // after a visualizer reconnect / refresh.
    apply();
})();

// --- Adaptive detector controls ---
// Auto-applied on change. Posts to /detect/apply which forwards to the dApp
// over the existing REQ/REP control socket. The dApp owns the detector state;
// any parameter change (including granularity) resets the noise-floor history.
(function() {
    const enEl  = document.getElementById('det-enabled');
    const grEl  = document.getElementById('det-granularity');
    const thrEl = document.getElementById('det-threshold-db');
    const histEl= document.getElementById('det-hist-depth');
    const embEl = document.getElementById('det-embargo-secs');
    const maskEl= document.getElementById('det-mask-sensing');
    const help  = document.getElementById('det-help');
    if (!enEl || !grEl || !thrEl || !histEl || !embEl) return;

    function updateHelp(reply) {
        if (!help) return;
        if (reply && reply.ok === false) {
            help.textContent = 'Error: ' + (reply.error || 'unknown');
            help.style.color = 'var(--red)';
            return;
        }
        if (reply && reply.ok && reply.enabled) {
            help.textContent = `active — ${reply.granularity}, ${reply.n_bins} bins, ` +
                               `thr=${reply.threshold_db} dB, hist=${reply.hist_depth}, ` +
                               `embargo=${reply.embargo_secs}s`;
            help.style.color = 'var(--green)';
        } else {
            help.textContent = 'disabled';
            help.style.color = 'var(--muted)';
        }
    }

    async function apply() {
        const cfg = {
            enabled: enEl.value === 'true',
            granularity: grEl.value,
            threshold_db: parseFloat(thrEl.value),
            hist_depth: parseInt(histEl.value, 10),
            embargo_secs: parseFloat(embEl.value),
            mask_with_sensing: maskEl ? (maskEl.value === 'true') : false,
        };
        // Guard against transient NaN while the user is typing.
        if (!Number.isFinite(cfg.threshold_db)) cfg.threshold_db = 20.0;
        if (!Number.isFinite(cfg.hist_depth) || cfg.hist_depth < 1) cfg.hist_depth = 32;
        if (!Number.isFinite(cfg.embargo_secs) || cfg.embargo_secs < 0) cfg.embargo_secs = 9.9;
        try {
            const r = await fetch('/detect/apply', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config: cfg }),
            }).then(x => x.json());
            updateHelp(r);
        } catch (_) { /* transient — next change will retry */ }
    }

    [enEl, grEl, thrEl, histEl, embEl, maskEl].filter(Boolean).forEach(el => el.addEventListener('change', apply));
    // Push initial state on load so the dApp matches the UI after refresh.
    apply();
})();

// --- Detector mask stats: count blocked bins per indication, surface in footer ---
(function() {
    const stat = document.getElementById('stat-blocked');
    if (!stat) return;
    let lastMs = 0;
    dataHub.subscribe(frame => {
        const now = performance.now();
        if (now - lastMs < 100) return;   // ~10/s — human-readable card, not a panel
        lastMs = now;
        if (!frame.detMask || !frame.detN) {
            stat.textContent = '–';
            return;
        }
        let blocked = 0;
        for (let i = 0; i < frame.detN; i++) if (frame.detMask[i]) blocked++;
        stat.textContent = blocked + ' / ' + frame.detN;
    });
})();

// --- Compute time card: rolling avg of kernel_us across recent indications ---
(function() {
    const el = document.getElementById('stat-compute');
    if (!el) return;
    const WIN = 64;     // sliding window
    const buf = [];
    let sum = 0;
    let lastMs = 0;
    dataHub.subscribe(frame => {
        const us = frame.kernelUs;
        if (us === undefined) return;
        buf.push(us);
        sum += us;
        if (buf.length > WIN) sum -= buf.shift();
        const now = performance.now();
        if (now - lastMs < 100) return;   // keep the rolling avg accurate, throttle the DOM write to ~10/s
        lastMs = now;
        el.firstChild.textContent = (sum / buf.length).toFixed(0) + ' ';
    });
})();

// --- Detector mask strip: draw a thin red bar above the waterfall ---
// Bins are placed across the full strip width regardless of detector
// granularity (273 PRBs or 3276 SCs), so the x-axis stays aligned with the
// waterfall below. A blocked bin (mask[i] != 0) becomes a red rectangle of
// width strip_width / detN.
(function() {
    const canvas = document.getElementById('mask-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let cssW = 0, cssH = 0;
    function resize() {
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        cssW = rect.width;  cssH = rect.height;
        canvas.width  = Math.max(1, Math.round(cssW * dpr));
        canvas.height = Math.max(1, Math.round(cssH * dpr));
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();
    window.addEventListener('resize', resize);

    // Idle render: empty strip with subtle baseline so it's visible even
    // before the first detector frame arrives.
    function clear() {
        ctx.clearRect(0, 0, cssW, cssH);
    }
    clear();

    // Cache the latest mask so a zoom change can redraw the strip immediately,
    // even while paused or between throttled frames.
    let lastMask = null, lastN = 0, lastMs = 0;

    // Draw the blocked bins, mapping each bin's full-band position through the
    // active freq zoom [xMin, xMax] so the strip stays column-aligned with the
    // (zoomed) waterfall below. Bin i spans normalized [i/n, (i+1)/n] of the full
    // subcarrier band; the visible window [xMin, xMax] is stretched to full width.
    function render() {
        // If layout hasn't settled (cssW==0 because the panel was hidden), retry.
        if (cssW === 0 || cssH === 0) resize();
        clear();
        if (!lastMask || lastN === 0) return;
        const lo = vizState.xMin, hi = vizState.xMax;
        const span = Math.max(1e-6, hi - lo);
        ctx.fillStyle = '#f85149';
        for (let i = 0; i < lastN; i++) {
            if (!lastMask[i]) continue;
            const xl = (((i / lastN) - lo) / span) * cssW;
            const xr = ((((i + 1) / lastN) - lo) / span) * cssW;
            if (xr <= 0 || xl >= cssW) continue;        // bin outside the zoom window
            const x = Math.floor(xl);
            const w = Math.max(1, Math.ceil(xr - xl));  // >=1px so a lone blocked bin stays visible
            ctx.fillRect(x, 0, w, cssH);
        }
    }

    dataHub.subscribe(frame => {
        const now = performance.now();
        if (now - lastMs < 100) return;   // ~10/s — block indicator, not a panel
        lastMs = now;
        lastMask = frame.detMask || null;
        lastN    = frame.detN || 0;
        render();
    });
    // Redraw immediately when the freq zoom changes (applyZoom dispatches this).
    window.addEventListener('vizzoom', render);
})();

// --- Sidebar toggle ---
(function() {
    const btn = document.getElementById('sidebar-toggle');
    if (!btn) return;
    btn.addEventListener('click', () => {
        document.body.classList.toggle('sidebar-closed');
        /* Let the canvas resize handlers pick up the new size. */
        window.dispatchEvent(new Event('resize'));
    });
})();

// --- Spectrum SM: PRB-block enable/disable (gNB control) ---
// On change → POST /block/apply {enabled}. On load → query-only (no `enabled`)
// so the toggle reflects the dApp's current state without forcing it off.
(function() {
    const el   = document.getElementById('blk-enabled');
    const help = document.getElementById('blk-help');
    if (!el) return;

    function reflect(enabled, ok, err) {
        if (typeof enabled === 'boolean') el.value = enabled ? 'true' : 'false';
        if (!help) return;
        if (ok === false) {
            help.textContent = 'error: ' + (err || 'unknown');
            help.style.color = 'var(--red)';
            return;
        }
        const on = el.value === 'true';
        help.textContent = on ? 'forwarding occupied PRBs to gNB (UL+DL)' : 'disabled';
        help.style.color = on ? 'var(--green)' : 'var(--muted)';
    }

    async function send(body) {
        try {
            const r = await fetch('/block/apply', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            }).then(x => x.json());
            reflect(r.enabled, r.ok, r.error);
        } catch (_) { /* transient — next change retries */ }
    }

    el.addEventListener('change', () => send({ enabled: el.value === 'true' }));
    send({});  // query-only on load → reflect dApp state, no side effect
})();

// --- Visualizer view controls: freq zoom + pause (client-side only) ---
(function() {
    const zMin  = document.getElementById('viz-zoom-min');
    const zMax  = document.getElementById('viz-zoom-max');
    const pause = document.getElementById('viz-pause-btn');
    const pHelp = document.getElementById('viz-pause-help');
    const xaxis = document.getElementById('wf-xaxis');
    const N_PRBS = __N_PRBS__, SC_PER_PRB = 12, N_SC = N_PRBS * SC_PER_PRB;

    function applyZoom() {
        let lo = parseInt(zMin && zMin.value, 10);
        let hi = parseInt(zMax && zMax.value, 10);
        if (!Number.isFinite(lo)) lo = 0;
        if (!Number.isFinite(hi)) hi = N_PRBS - 1;
        lo = Math.max(0, Math.min(N_PRBS - 1, lo));
        hi = Math.max(0, Math.min(N_PRBS - 1, hi));
        if (hi < lo) { const t = lo; lo = hi; hi = t; }
        vizState.xMin = (lo * SC_PER_PRB) / N_SC;
        vizState.xMax = ((hi + 1) * SC_PER_PRB) / N_SC;
        if (xaxis) {
            xaxis.textContent = (lo === 0 && hi === N_PRBS - 1)
                ? 'Subcarrier Index 0..3275 (273 PRBs x 12 SCs, antenna 0)'
                : ('PRB ' + lo + '..' + hi + ' (SC ' + (lo * SC_PER_PRB) + '..' +
                   ((hi + 1) * SC_PER_PRB - 1) + ', antenna 0)');
        }
        // Notify panels that aren't redrawn every frame (the blocked strip) to
        // re-render against the new zoom window immediately.
        window.dispatchEvent(new Event('vizzoom'));
    }
    if (zMin) zMin.addEventListener('change', applyZoom);
    if (zMax) zMax.addEventListener('change', applyZoom);
    applyZoom();

    // Slot / symbol display filters (CSV + ranges, e.g. "8,9" or "2-12"; empty = all).
    const slotsEl = document.getElementById('viz-slots');
    const symsEl  = document.getElementById('viz-symbols');
    function parseSet(s, lo, hi) {
        if (typeof s !== 'string' || s.trim() === '') return null;
        const out = new Set();
        for (const tok of s.split(',')) {
            const t = tok.trim();
            if (t === '') continue;
            const m = t.match(/^(\d+)\s*-\s*(\d+)$/);
            if (m) {
                let a = +m[1], b = +m[2]; if (a > b) { const x = a; a = b; b = x; }
                for (let v = a; v <= b; v++) if (v >= lo && v <= hi) out.add(v);
            } else if (/^\d+$/.test(t)) {
                const v = +t; if (v >= lo && v <= hi) out.add(v);
            }
        }
        return out.size ? out : null;
    }
    function applyFilters() {
        vizState.slots   = slotsEl ? parseSet(slotsEl.value, 0, 1023) : null;
        vizState.symbols = symsEl  ? parseSet(symsEl.value,  0, 13)   : null;
    }
    if (slotsEl) slotsEl.addEventListener('change', applyFilters);
    if (symsEl)  symsEl.addEventListener('change', applyFilters);
    applyFilters();

    if (pause) {
        pause.addEventListener('click', () => {
            vizState.paused = !vizState.paused;
            pause.innerHTML = vizState.paused ? '&#9654; Resume' : '&#10073;&#10073; Pause';
            pause.classList.toggle('rec-active', vizState.paused);
            if (pHelp) {
                pHelp.textContent = vizState.paused ? 'paused (display frozen)' : 'live';
                pHelp.style.color = vizState.paused ? 'var(--red)' : 'var(--muted)';
            }
        });
    }
})();
</script>
</body>
</html>
'''


# --- Recording control: proxy REQ to the dApp's REP socket. -----------------
# REQ sockets are not thread-safe and their state machine breaks on timeout —
# so we serialize with a lock and reset the socket if a recv ever fails.
_ctrl_lock = threading.Lock()
_ctrl_socket = None
_ctrl_endpoint = "tcp://localhost:5560"
# When None, no REP server owns the control port (e.g. the Python dApp doesn't
# bind one). We must NOT connect to 5560 then: on a shared host that is the
# co-resident C++ dApp's REP, so a /block/apply would send a real UL+DL PRB
# block to the gNB via the wrong agent. Disabled → routes report unavailable.
_ctrl_enabled = True


def _ctrl_request(payload, timeout_ms=2000):
    global _ctrl_socket
    if not _ctrl_enabled:
        return {"ok": False, "error": "control not available (no dApp REP bound)"}
    with _ctrl_lock:
        if _ctrl_socket is None:
            ctx = zmq.Context.instance()
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
            sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
            sock.connect(_ctrl_endpoint)
            _ctrl_socket = sock
        try:
            _ctrl_socket.send_string(json.dumps(payload))
            reply = _ctrl_socket.recv_string()
            return json.loads(reply)
        except zmq.error.Again:
            try:
                _ctrl_socket.close(0)
            except Exception:
                pass
            _ctrl_socket = None
            return {"ok": False, "error": "control timeout — is the dApp reachable on " + _ctrl_endpoint + "?"}
        except Exception as e:
            try:
                _ctrl_socket.close(0)
            except Exception:
                pass
            _ctrl_socket = None
            return {"ok": False, "error": f"control error: {e}"}


@app.route('/record/start', methods=['POST'])
def record_start():
    body = request.get_json(silent=True) or {}
    try:
        duration_s = float(body.get('duration_s', 10))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "duration_s must be a number"}), 400
    if duration_s <= 0 or duration_s > 86400:
        return jsonify({"ok": False, "error": "duration_s must be in (0, 86400]"}), 400
    # Free-form text — embedded into the .sigmf-meta as `dapp:user_notes`.
    notes = body.get('notes', '')
    if not isinstance(notes, str):
        return jsonify({"ok": False, "error": "notes must be a string"}), 400
    if len(notes) > 16 * 1024:
        return jsonify({"ok": False, "error": "notes too long (max 16 KiB)"}), 400
    payload = {"cmd": "start", "duration_s": duration_s}
    if notes:
        payload["notes"] = notes
    return jsonify(_ctrl_request(payload))


@app.route('/record/stop', methods=['POST'])
def record_stop():
    return jsonify(_ctrl_request({"cmd": "stop"}))


@app.route('/record/status', methods=['GET'])
def record_status():
    return jsonify(_ctrl_request({"cmd": "status"}))


@app.route('/viz/apply', methods=['POST'])
def viz_apply():
    """Forward visualizer publish controls (granularity + subsample) to the
    dApp. Body: {"config": {freq_gran, time_gran, subsample_n}}. Partial
    updates allowed — fields the user didn't change can be omitted."""
    body = request.get_json(silent=True) or {}
    config = body.get('config')
    if not isinstance(config, dict):
        return jsonify({"ok": False, "error": "missing 'config' object"}), 400
    return jsonify(_ctrl_request({"cmd": "viz", "config": config}))


@app.route('/detect/apply', methods=['POST'])
def detect_apply():
    """Forward AdaptiveThresholdDetector controls to the dApp. Body:
    {"config": {enabled, granularity, threshold_db, hist_depth, embargo_secs}}.
    Partial updates allowed — only fields present in config are applied.
    Any parameter change resets the detector's noise-floor history."""
    body = request.get_json(silent=True) or {}
    config = body.get('config')
    if not isinstance(config, dict):
        return jsonify({"ok": False, "error": "missing 'config' object"}), 400
    return jsonify(_ctrl_request({"cmd": "detect", "config": config}))


@app.route('/block/apply', methods=['POST'])
def block_apply():
    """Enable/disable gNB PRB-block forwarding (Spectrum SM, RF=1, control_id=1).
    Body: {"enabled": bool} to set, or {} to query without changing. Forwards
    {"cmd":"block"[, "enabled"]}; the dApp reply carries the current {"enabled"}."""
    body = request.get_json(silent=True) or {}
    payload = {"cmd": "block"}
    if 'enabled' in body:
        payload['enabled'] = bool(body['enabled'])
    return jsonify(_ctrl_request(payload))


@app.route('/clientstats', methods=['POST'])
def client_stats():
    """Browser reports its cumulative consumed/dropped frame counts (~1/s) so the
    smooth-rate ceiling is measurable server-side (the WS TCP buffer always drains,
    so server-side ws-push can't reveal browser-side drops). Logged to stdout."""
    d = request.get_json(silent=True) or {}
    sid = d.get('sid')
    if sid:                       # update pacing feedback (beacons carry sid+consumed[+paused])
        c = int(d.get('consumed', 0))
        p = bool(d.get('paused'))
        with _pace_lock:
            # (re)baseline so backlog -> 0 on the first beacon AND on resume (paused->active).
            # A hidden/paused tab stops consuming; without the resume rebase the in-flight
            # would stay pinned at the budget and the feed would never restart ("freeze").
            if sid not in _pace_consumed_base or (_pace_paused.get(sid) and not p):
                _pace_consumed_base[sid] = c - _pace_sent.get(sid, 0)
            _pace_paused[sid] = p
            _pace_consumed[sid] = c - _pace_consumed_base[sid]
    if d.get('worker_n') is not None:    # full perf POST (~1/s); skip log for the 250ms beacons
        print(f"[CLIENT] consumed={d.get('consumed',0)} dropped={d.get('dropped',0)} | "
              f"worker={d.get('worker_ms',0)}ms/{d.get('worker_n',0)} "
              f"main={d.get('main_ms',0)}ms/{d.get('main_n',0)} "
              f"dens={d.get('dens_ms',0)}ms/{d.get('dens_n',0)} "
              f"wf={d.get('wf_ms',0)}ms/{d.get('wf_n',0)}  (ms summed over ~1s = ms/s load per stage)", flush=True)
    return jsonify({"ok": True})


@app.route('/insi-logo')
def insi_logo():
    """Serve the INSI / Northeastern logo bundled in the visualizer dir."""
    logo_path = os.path.join(os.path.dirname(__file__), 'insi_white.png')
    if os.path.exists(logo_path):
        return send_file(logo_path, mimetype='image/png')
    return ('', 404)


@app.route('/')
def index():
    return render_template_string(
        HTML_TEMPLATE.replace('__N_STREAMS__', str(N_STREAMS))
                     .replace('__COMPRESS__', '1' if VIZ_COMPRESS else '0')
                     .replace('__N_SC__', str(VIZ_NUM_PRBS * VIZ_SC_PER_PRB))
                     .replace('__N_PRBS__', str(VIZ_NUM_PRBS)))


def _resolve_ssl_context(cert, key, script_dir):
    """Build an ssl_context for app.run() when serving HTTPS.

    Precedence: explicit --cert/--key; else a persistent self-signed pair next to
    this script (generated once via openssl, so the browser only has to trust it
    once); else Werkzeug's ephemeral 'adhoc' cert (requires the `cryptography`
    package, and re-warns every restart).
    """
    if cert and key:
        if os.path.exists(cert) and os.path.exists(key):
            print(f"[HTTPS] using provided cert/key: {cert}, {key}")
            return (cert, key)
        print(f"[HTTPS] --cert/--key not found ({cert}, {key}); using self-signed instead")

    gen_cert = os.path.join(script_dir, 'viz_selfsigned_cert.pem')
    gen_key  = os.path.join(script_dir, 'viz_selfsigned_key.pem')
    if os.path.exists(gen_cert) and os.path.exists(gen_key):
        print(f"[HTTPS] reusing self-signed cert: {gen_cert}")
        return (gen_cert, gen_key)
    try:
        subprocess.run(
            ['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
             '-keyout', gen_key, '-out', gen_cert, '-days', '3650',
             '-subj', '/CN=subcarrier-power-visualizer'],
            check=True, capture_output=True)
        print(f"[HTTPS] generated self-signed cert: {gen_cert} (browsers warn once)")
        return (gen_cert, gen_key)
    except Exception as e:
        print(f"[HTTPS] openssl self-signed generation failed ({e}); "
              f"falling back to Werkzeug 'adhoc' (needs the 'cryptography' package)")
        return 'adhoc'


def _serve(port, ssl_context, scheme):
    print(f"Serving dashboard on {scheme}://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True,
            use_reloader=False, ssl_context=ssl_context)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Subcarrier Power Visualizer")
    parser.add_argument('--port', type=int, default=5001,
                        help='HTTP port (0 disables HTTP; default: 5001)')
    parser.add_argument('--https-port', type=int, default=0,
                        help='HTTPS port (0 disables HTTPS; default: 0)')
    parser.add_argument('--cert', type=str, default='',
                        help='TLS certificate PEM for HTTPS (default: auto self-signed)')
    parser.add_argument('--key', type=str, default='',
                        help='TLS private key PEM for HTTPS (default: auto self-signed)')
    parser.add_argument('--zmq-port', type=int, default=5559, help='ZMQ subscriber port (default: 5559)')
    parser.add_argument('--ctrl-port', type=int, default=5560,
                        help='dApp recording control port (default: 5560)')
    parser.add_argument('--ctrl-host', type=str, default='localhost',
                        help='dApp recording control host (default: localhost)')
    parser.add_argument('--num-prbs', type=int, default=VIZ_NUM_PRBS,
                        help='Active carrier PRB count; sets the grid width to '
                             'num_prbs*12 subcarriers (default: %(default)s)')
    args = parser.parse_args()

    # Size the render grid to the active bandwidth before the page is served.
    VIZ_NUM_PRBS = args.num_prbs

    # --ctrl-port <= 0 disables the control proxy (no REP to talk to). This is
    # the default when the Python dApp spawns us, since it binds no REP and 5560
    # may belong to a co-resident C++ dApp.
    _ctrl_enabled = args.ctrl_port > 0
    _ctrl_endpoint = f"tcp://{args.ctrl_host}:{args.ctrl_port}"

    threading.Thread(target=zmq_receiver, args=(args.zmq_port,), daemon=True).start()

    # The browser JS already selects ws:// vs wss:// from the page protocol, so
    # the same app serves WebSockets correctly on both schemes. app.run() blocks,
    # so when both HTTP and HTTPS are enabled the first listener runs on a daemon
    # thread and the last on the main thread.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    listeners = []
    if args.port and args.port > 0:
        listeners.append((args.port, None, 'http'))
    if args.https_port and args.https_port > 0:
        listeners.append((args.https_port,
                          _resolve_ssl_context(args.cert, args.key, script_dir),
                          'https'))
    if not listeners:
        raise SystemExit("Nothing to serve: --port and --https-port are both 0.")

    print("Open the dashboard in your browser:")
    for port, _ctx, scheme in listeners:
        print(f"  {scheme}://localhost:{port}")

    for port, ctx, scheme in listeners[:-1]:
        threading.Thread(target=_serve, args=(port, ctx, scheme), daemon=True).start()
    _serve(*listeners[-1])
