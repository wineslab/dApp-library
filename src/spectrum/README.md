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
   └► ZMQ PUB (per-slot u8 power) ─► subcarrier_visualizer.py ─► browser (WebGL waterfall)
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
`visualization/subcarrier/subcarrier_visualizer.py` (Flask + WebSocket + WebGL2)
on `--viz-web-port` (5001). The dApp publishes one multipart frame per slot
(`subcarrier_power|sfn|slot|n_ant|n_sym|n_sc|u8|db_min|db_max|...` + u8 power);
see `visualization/subcarrier_pub.py`. Use `--external-viz` to feed an
already-running visualizer.

## Robustness

- Both shm readers detect a gNB restart (the gNB recreates the regions with a
  new inode) and remap.
- The FH staleness bound is derived from the ring depth in the shm header
  (`num_fh_rows`), not a hardcoded value.
