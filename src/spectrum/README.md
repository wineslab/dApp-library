# Spectrum dApp pipeline — from gNB radio to browser waterfall

The spectrum dApp senses the uplink, optionally blacklists PRBs on the gNB, and
streams a live subcarrier-power waterfall to a browser. It talks to the gNB over
E3 (libe3) and reads bulk IQ / sensing ranges from POSIX shared memory.

## Data flow

```
gNB PHY ── L1-KPM SM (RF=2) ──► E3 indication: shm pointer + validSymbolMask
   │                              └► dApp reads IQ from /e3_ran_buffers (FP16)
   └─ Spectrum SM (RF=1) ──► E3 indication: shm ref (shmWriteIdx, nRanges)
                                  └► dApp reads ranges from /e3_l2_sensing ring
dApp: magnitude ─► detector (masked adaptive noise floor) ─► PRB blacklist control (RF=1)
   └► ZMQ PUB (per-slot u8 power) ─► subcarrier_visualizer.py ─► browser (WebGL2 waterfall)
```

## Service models (both ASN.1/APER and JSON)

- **RF=2 L1-KPM** — per-slot IQ telemetry. The indication carries an shm
  reference (`/e3_ran_buffers`, `fhBufferIndex`, `fhWriteIndex`), `sfn`/`slot`,
  and `validSymbolMask` (14-bit UL-symbol bitmap). IQ layout is
  `[ant=4][sym=14][prb=273][sc=12]` FP16. See `e3_ran_buffers_reader.py`.
- **RF=1 Spectrum** — sensing-range telemetry + PRB-block / sensing-policy
  control. The `Spectrum-SensingIndication` references the `/e3_l2_sensing` ring
  (`shmWriteIdx`, `nRanges`); each record is a `(start_symbol, num_symbols,
  rb_start, rb_size)` rectangle. See `e3_l2_sensing_reader.py`.

The ASN.1 grammars (`defs/e3sm_oai_l1_kpm.asn`, `defs/e3sm_spectrum.asn`) are
copied **verbatim** from the OAI gNB so the wire stays byte-identical.

## Sensing window

The dApp caches the RF=1 sensing ranges by `(sfn, slot)` and applies them to the
matching RF=2 IQ slot (`sensing_only`, default on) to build a 2D (symbol, PRB)
keep-mask. Out-of-window columns are excluded from the adaptive noise-floor
history (not fed as zeros), so ambient UE PRBs don't skew the floor.

## Visualizer

`--demo-gui` binds a ZMQ PUB and (by default) spawns the standalone
`visualization/subcarrier/subcarrier_visualizer.py` (Flask + WebSocket + WebGL2,
R8 textures via `texStorage2D`) on `--viz-web-port` (5001). The dApp publishes one
multipart frame per slot (`subcarrier_power|sfn|slot|n_ant|n_sym|n_sc|u8|db_min|
db_max|det=…|det_n=…|det_thr=…` + u8 power, plus a detection-mask frame when the
detector has flagged PRBs); see `visualization/subcarrier_pub.py`. The render
grid width follows the active carrier (`--num-prbs`×12 subcarriers, e.g. 1272 for
106 PRB), so the waterfall renders on any bandwidth. Use `--external-viz` to feed
an already-running visualizer. The dApp tells it the carrier centre and
subcarrier spacing, so the x axis is labelled in MHz (ticks above the waterfall,
and a PRB / subcarrier / MHz readout under the cursor); without `--center-freq`
information the axis shows indices only.

## Dashboard control (protocol v1)

With `--demo-gui` the dApp also serves the dashboard's control panels on a ZMQ
REP socket, `tcp://127.0.0.1:5570` by default. `--ctrl-port N` picks another
port, `--ctrl-port 0` turns it off. The default port is not 5560 so it never
collides with another dApp's control port; if it is already taken the dApp warns
and runs without control (panels greyed), whereas an explicit `--ctrl-port` that
cannot be bound stops the dApp. With `--external-viz`, start the visualizer with
the same `--ctrl-port` (and `--num-prbs`).

Requests are one JSON object per frame with a `"cmd"`; every reply has `"ok"`
and, on failure, `"error"`. The dashboard waits 2 s for the reply.

| cmd | request | effect |
|---|---|---|
| `caps` | – | what this dApp supports: detector kind, per-field availability and the reason when a field is greyed out |
| `block` | `enabled` (bool); none = query | start/stop forwarding detections to the gNB; xApp-sourced blocks stay installed |
| `detect` | `config`: `enabled`, `granularity` (`sc`/`prb`), `threshold_db`, `hist_depth`, `embargo_secs`, `mask_with_sensing`; `{}` = query | change the active detector; all-or-nothing |
| `viz` | `config`: `freq_gran`, `time_gran`, `agg`, `subsample_n`; `{}` = query | what the dApp publishes to the dashboard |
| `start` / `stop` / `status` | `duration_s` (0, 86400], `notes` | SigMF recording |

`mask_with_sensing` (the L1xL2 filter, `sensing_only`) masks the dashboard as well
as the detector input: each slot's keep mask (sensing ranges and valid symbols)
is built once and the published frame has every other cell at the floor, so the
dashboard shows what the detector sees, slot by slot, without the UE's own
transmissions. A slot with no sensed cell is not published.

`detect` applies to the threshold detectors. Under `--policy` the policy decides
the blocks, so only `mask_with_sensing` (the L1xL2 filter, i.e. `sensing_only`)
can be changed; the other fields are greyed out. A static detector has no
embargo, and its `hist_depth` is the averaging window. Granularity `prb`
averages the 12 subcarriers of a PRB before detection and rebuilds the detector
(history is lost); the red strip still shows the result per subcarrier. A change
that resets the adaptive history (`hist_depth`, `granularity`) while a constant
incumbent is already on makes that incumbent the new noise floor, so it is no
longer detected; the adaptive floor suits bursty interference.

Recording (`start`) is independent of `--save-iqs`: it writes every antenna,
every symbol and the gNB's whole 273-PRB slot as `cf32_le`, one record per slot
laid out `[ant][sym][prb][sc][I,Q]` (PRBs beyond the cell bandwidth are zero),
with one annotation per slot (`sfn`, `slot`, sensing ranges, valid-symbol mask).
A full 4-antenna slot is about 1.8 MB, so the writer runs on its own thread and
reports `dropped` slots instead of stalling detection; `start` is refused with
less than 2 GiB free in the log directory.

## Robustness

- Both shm readers detect a gNB restart (the gNB recreates the regions with a
  new inode) and remap.
- The FH staleness bound is derived from the ring depth in the shm header
  (`num_fh_rows`), not a hardcoded value.
