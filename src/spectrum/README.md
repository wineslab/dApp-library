# Spectrum dApp pipeline — from gNB radio to browser waterfall

This document is the **one-place explanation** of how raw IQ samples captured
by an OpenAirInterface (OAI) gNB end up as a live waterfall on a web
dashboard. Read it front-to-back if you have never touched this code; jump to
the section you need if you have.

The pipeline crosses three processes (gNB, dApp, browser) and four threads
inside the dApp. The cost of getting one of the stages wrong is silent data
loss, so the design is built around two ideas:

1. **Read the data as early as possible**, before any decision that could
   delay it. The data lives in a small ring in shared memory that the gNB
   overwrites every few slots.
2. **Filter, decimate, and quantise downstream**, where it is safe to drop
   information — never on the inbound path.

---

## 1. The problem in one paragraph

A 5G NR uplink slot is a 14-symbol × N-subcarrier block of complex samples
("IQ"). A *spectrum sensing* dApp wants to look at those samples to see who
else is using the band (other operators, jammers, etc.) and tell the gNB
which PRBs to avoid. The gNB has to ship each slot of raw rxdataF out to
the dApp fast enough not to disturb the radio thread, and the dApp has to
read, process, display, and (optionally) feed a PRB-block control back —
without ever back-pressuring the gNB.

> **Note on the upstream sensing-symbols variant.** That codebase adds a
> per-slot `scheduled_symbol_mask` so the dApp can isolate the symbols
> the scheduler left unallocated. This build deliberately omits that
> field (the scheduler-side OR-of-TDA-spans is not ported). The dApp
> sees the full slot of IQ and treats every symbol the same; the only
> surviving feedback channel is the PRB-block control.

---

## 2. Bird's-eye view

```
┌─────────────────────── gNB process ───────────────────────┐
│                                                            │
│  PHY (L1 RX) ─────────► spectrum_shm_push_rxdataF()        │
│                          ─► writes ring entry under seqlock│
│                          ─► signals SM worker via cond-var │
│                                                            │
│  spectrum_shm  ───────────►  /dev/shm/oai_e3_rxdataF_<pid> │
│   (ring + meta + IQ)         (POSIX shared memory)         │
│                                                            │
│  E3 SM worker ────────────►  small "IQ-indication" msg     │
│                              (envelope + meta pointer)     │
└──────────────────────────┬─────────────────────────────────┘
                           │ E3AP over ZMQ/IPC
                           ▼
┌─────────────────────── dApp process ───────────────────────┐
│                                                            │
│ inbound thread (libe3)                                     │
│   _handle_indication() ──► decode envelope                 │
│                          ─► parse SpectrumShmMeta          │
│                          ─► account seqno / drops          │
│                          ─► seqlock-read IQ from shm       │
│                          ─► enqueue (queue, depth=4)       │
│                                                            │
│ worker thread                                              │
│   _indication_worker() ──► mag = √(I² + Q²)                │
│                          ─► dApp detector (optional)       │
│                          ─► dashboard.process_iq_data(...) │
│                                                            │
│ dashboard emit-loop thread (~60 Hz)                        │
│                          ─► dB conversion                  │
│                          ─► active-band slice              │
│                          ─► row aggregation (mean-fold)    │
│                          ─► uint16 quantisation            │
│                          ─► K-in-flight ack gate           │
│                          ─► socket.io binary frame ───┐    │
└───────────────────────────────────────────────────────┼────┘
                                                        │
                                                        ▼
                ┌──────────── browser ──────────────────────┐
                │                                            │
                │ socket.io main thread                      │
                │   ─► transfer ArrayBuffer to worker        │
                │                                            │
                │ OffscreenCanvas worker                     │
                │   ─► uint16 → dB (× scale + bias)          │
                │   ─► palette LUT lookup → RGBA             │
                │   ─► OffscreenCanvas.transferToImageBitmap │
                │   ─► ack ("client_done") back to server    │
                └────────────────────────────────────────────┘
```

Every arrow that crosses a process boundary is a potential drop point; every
arrow that crosses a thread boundary inside the dApp is a potential reorder
or back-pressure point. The rest of this document explains how each crossing
is handled.

---

## 3. The shared-memory wire format

The gNB and dApp share one POSIX shared-memory segment per gNB instance,
named `/dev/shm/oai_e3_rxdataF_<pid>`. The C side is defined in
`openair2/E3AP/service_models/spectrum_sm/spectrum_shm.h`; the Python
mirror is in `spectrum_shm_reader.py`.

The segment contains:

```
┌──────────────────────────────────────────────┐
│ spectrum_shm_header_t      (≈ 80 bytes)      │
│   magic / version / dimensions / offsets     │
├──────────────────────────────────────────────┤
│ producer state                               │
│   newest write index, etc.                   │
├──────────────────────────────────────────────┤
│ ring of N entries:                           │
│  ┌────────────────────────────────────────┐  │
│  │ spectrum_shm_entry_header_t (64 bytes) │  │
│  │   seqlock | sfn | slot | direction     │  │
│  │   n_antennas | timestamp_ns            │  │
│  │   entry_seqno | reserved[6]            │  │
│  ├────────────────────────────────────────┤  │
│  │ payload: rxdataF for the whole slot    │  │
│  │   shape (n_beams, ant_max, 14 × N, 2)  │  │
│  │   c16_t = pair of int16                │  │
│  └────────────────────────────────────────┘  │
│  ...                                         │
└──────────────────────────────────────────────┘
```

The E3 *IndicationMessage* carries only a small metadata blob
(`SpectrumShmMeta`, **80 bytes**), not the IQ itself. The IQ stays in
shared memory and is read directly by the dApp.

The wire-format version is **`SPECTRUM_SHM_VERSION = 1`** in this build.
The upstream sensing-symbols variant defines a v2 layout that adds a
4-byte `scheduled_symbol_mask` after `direction` in both the entry
header and the indication metadata (84-byte meta in v2). This build
omits that field because the scheduler-side population is not ported;
see §1.

### Why a separate ring?

5G NR UL slots arrive every 250 µs to 1 ms. A full slot of IQ is large
(≈ 380 KB at 106 PRBs, 4 antennas), so copying it through the E3 socket
would saturate the link. Shared memory lets the gNB drop the IQ into a
fixed location and only ship the dApp a tiny pointer message.

### Why a seqlock?

The gNB is the single writer, multiple readers may exist. Locks would block
the radio thread. A seqlock lets the reader detect torn reads with two
counter snapshots (one before, one after) without blocking the writer.
See `SpectrumShmReader.read_entry()` in `spectrum_shm_reader.py:162`.

### What was removed: per-symbol TDA mask

In the upstream sensing-symbols variant, the gNB MAC scheduler stamps a
14-bit `scheduled_symbol_mask` at `commit_ul_alloc` time (one bit per
OFDM symbol of the slot; bit *i* = "symbol *i* was scheduled for UE
traffic"). The dApp would then slice `mag_batch[free_syms]` and ship
only the unscheduled rows, optionally dropping fully-scheduled slots.

This build does not implement that path:

* `spectrum_shm.h` here has no `scheduled_symbol_mask` field in either
  the entry header or the indication metadata.
* The MAC-side OR-into-mask logic (`nr_ul_schedule` clear +
  `commit_ul_alloc` OR) is not ported.
* The dApp's `_tally_sensing_window`, `_sensing_window_*` counters,
  `sensing_only` flag, and per-symbol slice are removed.

The dApp therefore always treats the slot as 14 fully-available
symbols. Coarser-grained spectrum control still works through the
PRB-block control path described in §4.3.

---

## 4. Stage-by-stage walkthrough

### 4.1 gNB-side: publish

File (on the OAI gNB side):

* `openair2/E3AP/service_models/spectrum_sm/spectrum_shm.c` —
  `spectrum_shm_push_rxdataF()` is invoked from
  `phy_procedures_nr_gNB.c` on every UL slot. It lazily creates the
  shm segment on first call (the PHY dimensions aren't known until
  L1 runs), then writes the full slot into the next ring entry under
  a seqlock and signals the SM worker thread via a cond-var.

The SM worker (`spectrum_sm.c::spectrum_sm_thread_main`) drains
recently-pushed seqnos, encodes one `Spectrum-IndicationData` envelope
per slot, and emits it to each subscribed dApp. The envelope's
`iqDataIndication.iqSamples` field carries the 80-byte
`SpectrumShmMeta` blob (NOT the IQ itself).

### 4.2 dApp inbound thread: `_handle_indication()`

File: `spectrum_dapp.py`, around line 880.

Runs on the libe3 callback thread. Its job is **never to block**: any
delay here is paid in dropped slots, because the ring keeps moving.

```
recv → decode envelope → parse meta → track seqno gap
                                   → read IQ from shm        ← critical
                                   → enqueue for worker
```

Important details:

* **Reading the IQ happens inline on this thread**, not deferred. By the
  time the worker thread sees the item, the underlying ring entry may
  already have been overwritten. A copy into a private numpy array is
  unavoidable; we just want to do it as soon as the indication lands.
* **The worker queue is `maxsize=4`** and uses *drop-oldest* on overflow.
  Worker lag delays display; it does NOT delay or lose shm reads.

### 4.3 dApp worker thread: `_process_indication()`

File: `spectrum_dapp.py`, around line 989.

Runs the heavy work that the inbound thread cannot afford:

1. **Magnitude.** `mag = hypot(I, Q)` over the whole `(n_sym, fft_size)`
   batch. Pre-allocated float32 buffers — no per-frame allocation.
2. **Dashboard feed.** `self.demo.process_iq_data(("iq_mag_batch", ...))`
   queues the full magnitude batch (all 14 rows) into the dashboard's
   pending buffer. No per-symbol filter is applied here — see §1.
3. **Detector (every 4th slot).** Static or adaptive threshold detection
   over the FFT-shifted magnitude. If PRBs are flagged, an E3 report is
   sent to the gNB, optionally a PRB-block control as well. The block
   travels as a `Spectrum-PRBBlacklistControl` inside a
   `Spectrum-DAppControlData` envelope; the gNB's SM dispatcher converts
   the list into a per-PRB 14-symbol bitmap and calls
   `set_prb_block_mask(mac, PRB_BLOCK_DIR_UL, mask, MAX_BWP_SIZE)`. The
   bitmap is OR'd into `vrb_map_UL` at slot start, so every downstream
   UL scheduling step (PRACH/PUCCH/data/sensing) treats the blocked
   PRBs as occupied.
4. **Latency record.** All `Timings` fields are converted to per-stage µs
   durations and appended to rolling 4096-sample deques.

The detector running 4× less often than the dashboard feed is a deliberate
trade: detection is expensive and infrequent decisions are fine; visual
feedback should stay smooth.

### 4.4 Dashboard server: `dashboard.py`

The dashboard has three jobs:

1. Receive numpy rows from the worker thread (`process_iq_data`).
2. Convert them to the smallest possible wire form.
3. Ship them to the browser as fast as the browser can absorb — and not
   faster.

It does this with a separate **emit-loop thread** running at ~60 Hz. The
producer side just appends `(row_count, uint16_bytes)` tuples into
`_pending_mag_rows`. The emit loop drains them under a lock.

#### Wire format

Each frame on the socket is:

```
[ Float64 age_us | row0 row1 ... rowN-1 ]   each row = N_bins × uint16
```

Quantisation: `uint16 = (db − QUANT_MIN_DB) × QUANT_SCALE`, clipped to
`[0, 65535]`. With `QUANT_MIN_DB = -20`, `QUANT_MAX_DB = 100` this is
≈ 1.83 mdB per LSB — invisible on any colour palette.

Float32 → uint16 halves wire bytes; the browser worker undoes the
quantisation inline as part of its palette-LUT multiply.

#### Backpressure: K-in-flight + watchdog

A naïve "emit and forget" design would let the browser fall behind
silently — engineio's queue grows on the server, eats RAM, and stalls.
Instead, we keep an `_inflight_ts` deque of up to **MAX_INFLIGHT = 3**
emits that have not yet been acked (`client_done` event from the browser
worker). The 4th emit is deferred until an ack lands. A 500 ms watchdog
retires unacked entries so a tab-close cannot pin the dApp forever.

#### Coalesce-on-overflow

If the producer rate exceeds the wire rate, `_pending_mag_rows` grows.
When it crosses **MAX_PENDING_MAG_ROWS = 128**, we **mean-fold the oldest
two entries into one** (`_coalesce_oldest_two_locked`). Information is
preserved as a time-average; the queue stays bounded; nothing is dropped
until a single entry alone would exceed the cap (very defensive
fallback).

#### Visibility gate

When the browser tab is hidden (`document.visibilityState === "hidden"`),
the browser worker emits a `set_client_visible(false)` event. The server
then:

* Drops the pending rows already queued (the browser will not drain them).
* Suppresses further emits until visible again.
* Resets `_inflight_ts` on re-show, so we don't wait for acks that will
  never come.

This was added because background-tab dashboards accumulated engineio
queue depth until the dApp ran out of memory.

### 4.5 Browser

The HTML page (`src/visualization/templates/index.html`) does very little
on the main thread: it owns the socket.io connection and immediately
hands every binary frame to a Web Worker via `transferControlToOffscreen`.

The worker:

1. Reads the age header (Float64) and the uint16 payload.
2. Multiplies uint16 → dB inline with the palette LUT (single pass).
3. Draws each row onto the OffscreenCanvas.
4. Calls `transferToImageBitmap` and posts it back to the main thread.
5. After paint completes, posts `client_done` back to the server (the
   K-in-flight ack).

Sidebar controls live on the main thread; they push `set_sampling_threshold`
and `set_aggregation_size` events to the server, which validate and
broadcast them back so all connected clients stay in sync.

---

## 5. Filtering: what's here, what isn't

This build has **no per-symbol filter**. Every slot's full 14-row
magnitude batch is forwarded to the dashboard and to the detector. The
upstream sensing-symbols variant ran a per-OFDM-symbol filter driven by
the gNB scheduler's `scheduled_symbol_mask` (see §3.3); that path is
intentionally absent here because the scheduler-side mask population is
not ported.

The only filter the dApp can apply to the gNB is the **PRB-block
control**:

* The detector flags PRBs whose energy exceeds threshold.
* The dApp encodes a `Spectrum-PRBBlacklistControl` (list of PRB
  indices) inside a `Spectrum-DAppControlData` envelope and sends it
  over E3.
* The gNB's SM handler converts the list into a per-PRB symbol bitmap
  (0x3FFF for each blocked PRB) and calls `set_prb_block_mask()` into
  the per-MAC `prb_block_state_t`.
* On every UL slot `apply_prb_block_masks()` (called at the top of
  `gNB_dlsch_ulsch_scheduler`, under `sched_lock`) OR's that bitmap
  into the current-slot `vrb_map_UL`; downstream UL scheduling
  (PRACH/PUCCH/SRS/data) consults `vrb_map_UL` and naturally skips
  PRBs marked occupied.

PRB granularity means a flagged PRB is excluded across all 14 symbols
of every subsequent slot until the block list is updated. It's coarser
than the upstream per-symbol filter but trivially deployable here
because no MAC-scheduler refactor is required.

---

## 6. Queues, drops, and how they are counted

Three counters distinguish what was lost where:

| Counter | When it goes up | Meaning |
|---|---|---|
| `wire_drops` | `entry_seqno` increment skipped a number | gNB pushed a slot we never received an indication for (E3 link or kernel buffer overflow) |
| `read_drops` | shm seqlock confirmed `entry_seqno` no longer matches | dApp inbound thread did not get to the ring entry in time; producer wrapped past us |
| `_ind_queue_dropped` | worker queue full | inbound thread fully read the slot but the worker is behind — oldest is dropped, freshest kept |

The first two are recoverable only by faster scheduling / bigger ring;
the third is recoverable by making the worker faster (mag, detector,
dashboard feed). Dashboard-side drops (`server_dropped`) are a separate
counter; they should be near zero in steady state thanks to coalesce.

---

## 7. Telemetry

Every 5 seconds, the dApp prints two `[LATENCY]` lines summarising rolling
per-stage µs durations (mean, p50, p99, max). The stages are:

```
producer_ts ──► recv ──► decode ──► meta ──► shm ──► qwait ──► mag ──► dash ──► det
        e2e    decode    meta      shm     qwait     mag     dash     det
                                                  └── work_total ──┘
```

`e2e` is the gNB-stamp → dApp-recv delta and shows total inbound latency.
`work_total` is `dequeue_ns → done_ns` and shows the worker-thread
budget.

The dashboard prints a separate `[DASHBOARD]` line every 5 s with:

* `submitted` / `emitted` / `coalesced` / `server_dropped` row counters
* `tick_hz` (emit-loop runs) / `call_hz` (actual socket emits)
* `mean_us` / `max_us` per emit, mean / max batch size

If `call_hz < tick_hz`, the K-in-flight gate or visibility gate is
holding the loop. If `coalesced > 0` consistently, the producer is faster
than the browser; either acceptable (data is averaged, not lost) or a
sign to drop a frame in the producer.

---

## 8. Where things live

```
src/spectrum/
  spectrum_dapp.py           # SpectrumSharingDApp class, all 3 threads
  spectrum_shm_reader.py     # POSIX-shm reader, seqlock retry, struct layout
  threshold_detector.py      # Static + Adaptive detector strategies
  adaptive_noise_floor.py    # Per-bin median noise floor for adaptive mode
  embargo_manager.py         # PRB embargo bookkeeping (adaptive mode)
  defs/
    e3sm_spectrum.asn        # ASN.1 grammar (envelope + payload variants)
    e3sm_spectrum.json       # JSON-schema mirror

src/visualization/
  dashboard.py               # Flask + SocketIO server, emit-loop, gates
  templates/index.html       # Page + OffscreenCanvas worker (inline)
  energy.py                  # Optional matplotlib energy plotter (legacy)
  iq.py                      # Optional matplotlib IQ plotter (legacy)

examples/
  spectrum_dapp.py           # Reference launcher (CLI args)
```

On the OAI gNB side (separate repo / build):

```
openair2/E3AP/service_models/spectrum_sm/
  spectrum_sm.c              # SM dispatcher: prbBlock (ctl=1), sensingPolicy (ctl=2)
                             #   prbBlock      -> set_prb_block_mask()
                             #   sensingPolicy -> set_sensing_policy()
  spectrum_enc.c / spectrum_dec.c  # E3 envelope encode/decode (ASN.1 or JSON)
  MESSAGES/ASN1/V1/e3sm_spectrum.asn  # source of truth for ASN.1

openair2/LAYER2/NR_MAC_gNB/
  gNB_scheduler_prb_block.c  # set_prb_block_mask + apply_prb_block_masks
                             #   per-MAC prb_block_state_t (DL/UL masks)
                             #   apply_prb_block_masks OR's into vrb_map_UL
                             #   at the top of every slot pass.
  gNB_scheduler.c            # gNB_dlsch_ulsch_scheduler: calls
                             #   apply_prb_block_masks() before any
                             #   scheduling step inspects vrb_map_UL.
```

---

## 9. Glossary

* **PRB** — Physical Resource Block, 12 contiguous subcarriers. The dApp
  works in PRB units when talking to the scheduler (block control, reports),
  and in raw FFT-bin units when computing magnitude.
* **PUSCH** — Physical Uplink Shared Channel. The actual UE-to-gNB
  uplink traffic.
* **TDA** — Time-Domain Allocation. The contiguous span of symbols in a
  slot assigned to one PUSCH transmission. Used by the upstream
  sensing-symbols variant to build the per-slot symbol mask; not
  exposed to the dApp in this build.
* **Slot** — 14 OFDM symbols. With 30 kHz SCS (FR1) a slot is 0.5 ms.
* **rxdataF** — OAI's name for received frequency-domain samples (post
  FFT, pre-equalisation). What ends up in the shared-memory payload.
* **Seqlock** — Lock-free synchronisation pattern: writer increments a
  counter to an odd value, writes, increments to the next even value;
  reader snapshots the counter before and after a read and retries if
  they differ or are odd.
* **E3 / E3SM** — Application-protocol interface between the gNB and the
  dApp. We use one Service Model (spectrum) over an IPC ZMQ link.
* **PRB block** — A list of PRB indices the dApp asks the gNB scheduler
  to avoid. Carried as `Spectrum-PRBBlacklistControl` over E3 and applied via
  `set_prb_block_mask()` into the per-MAC `prb_block_state_t`, which is
  OR'd into `vrb_map_UL` at slot start.  Coarser than the upstream
  per-symbol mask, but the only PRB-level feedback channel in this build.
* **K-in-flight** — Flow control where up to K frames can be outstanding
  without an ack before the sender pauses. Smooths over RTT jitter
  without unbounded queueing.
* **Coalesce-on-overflow** — When the queue exceeds a cap, the oldest
  two entries are mean-folded into one. Bounded memory, lossless on
  long-term average.

---

## 10. Reading order if you're new

1. This document.
2. `examples/spectrum_dapp.py` to see how the pieces are wired up.
3. `spectrum_shm_reader.py` (small, well-commented) to understand the
   wire format.
4. `spectrum_dapp.py::_handle_indication` and `_process_indication`.
5. `dashboard.py::_emit_loop` and `_take_pending`.
6. The browser worker inside `templates/index.html`.

If you only have time for one function, read `_handle_indication`. Every
other function exists to support what that function decides per slot.
