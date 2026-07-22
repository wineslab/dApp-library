#!/usr/bin/env python3
"""
dApp for Spectrum Sharing
"""

__author__ = "Andrea Lacava"

import threading
from collections import deque, OrderedDict
from dataclasses import dataclass
import queue
import time
import os
import json
from typing import override
import numpy as np
import asn1tools
import jsonschema
# np.set_printoptions(threshold=sys.maxsize)

from dapp.dapp import DApp
from e3interface.e3_encoder import JsonE3Encoder
from e3interface.e3_logging import dapp_logger, LOG_DIR
from spectrum.e3_ran_buffers_reader import (
    E3RanBuffersReader,
    SlotPointer,
    N_SC_PER_SLOT,
)
from spectrum.e3_l2_sensing_reader import E3L2SensingReader, SensingRange
from spectrum.threshold_detector import (
    ThresholdDetector,
    StaticThresholdDetector,
    AdaptiveThresholdDetector,
)


# Version of the spear-lake `dapp:` SigMF field list this dApp targets, emitted
# as dapp:schema_version in the SigMF global block. Co-versioned with
# spear-lake/docs/DAPP_SIGMF_FIELDS.yml (schema 1.0.0), which consolidated the
# ecosystem on the single `dapp:` namespace and retired the legacy `spear:` one.
#
# Known, accepted conformance gap (tracked in #82): two required 1.0.0 fields are
# not yet fully satisfied and are deferred to that issue —
#   - `dapp:iq_sample_count_start` is not written (needs a session-global,
#     non-resetting sample counter), and
#   - `dapp:effective_sample_rate` currently lands in the global block rather than
#     per-capture-segment (needs the libiqsaver per-segment metadata passthrough).
# We knowingly keep the 1.0.0 stamp until #82 lands rather than churn the version.
DAPP_SIGMF_SCHEMA_VERSION = "1.0.0"

# OFDM symbols per slot in 5G NR with normal cyclic prefix. Describes the time
# axis of the underlying resource grid (dapp:n_symbols). The on-disk record is
# one symbol's worth of frequency-domain FFT bins, so this is RAN geometry, not
# an on-disk reshape dimension.
NUM_OFDM_SYMBOLS_PER_SLOT = 14


@dataclass
class Timings:
    """Per-slot timestamps captured at each pipeline stage. Filled in by
    _handle_indication on the inbound thread, then by _process_indication
    on the worker thread, then read by _record_latency. All ns from
    time.monotonic_ns(); producer_ts_ns is the gNB's monotonic stamp from
    shm metadata (0 = unknown)."""
    recv_ns: int = 0
    decoded_ns: int = 0
    after_meta_ns: int = 0
    after_shm_ns: int = 0
    producer_ts_ns: int = 0
    dequeue_ns: int = 0
    after_mag_ns: int = 0
    after_dash_ns: int = 0
    after_det_ns: int = 0  # 0 = detector path didn't run this slot


def compute_fft_size(num_prbs: int, e_sampling: bool = False) -> int:
    """Return the per-symbol carrier count the dApp processes.

    Now backed by /e3_ran_buffers (cuBB layout) which presents carriers
    in natural PRB×SC order with no FFT-shift wraparound. So "fft_size"
    in the dApp is simply ``num_prbs * 12``; legacy power-of-2 padding
    and the USRP-E 3/4 scaling are no longer applicable. ``e_sampling``
    is accepted for backward compat with older example scripts that
    still pass it.
    """
    _ = e_sampling
    return num_prbs * 12


# ---------------------------------------------------------------------- #
# Sensing-policy mask helpers (used with SpectrumSharingDApp.send_sensing_policy
# and set_sensing_policy_logic).  Each helper returns a per-slot list of
# 14-bit symbol bitmaps suitable for the gNB's masked UL TDA selector --
# bit s set in mask[k] => "for slot k of the frame, prefer TDAs that
# don't claim symbol s, so symbol s stays free for sensing".
# ---------------------------------------------------------------------- #
def make_uniform_mask(n_slots: int, symbol_bitmap: int) -> list[int]:
    """Same 14-bit ``symbol_bitmap`` on every one of ``n_slots`` slots.

    Raises ValueError on out-of-range ``symbol_bitmap`` rather than
    silently truncating to 14 bits.  Matches the validation behaviour
    of ``make_periodic_toggle_callback`` and ``send_sensing_policy``.
    """
    if n_slots <= 0:
        raise ValueError(f"n_slots must be positive, got {n_slots}")
    if symbol_bitmap < 0 or symbol_bitmap > 0x3FFF:
        raise ValueError(
            f"symbol_bitmap={symbol_bitmap:#x} not a valid 14-bit mask "
            f"(0..0x3FFF)"
        )
    return [symbol_bitmap] * n_slots


def make_symbol_range_mask(n_slots: int, start_sym: int, num_sym: int,
                           target_slots: list[int] | None = None) -> list[int]:
    """Mask reserving symbols [start_sym, start_sym+num_sym) on each slot.

    ``target_slots`` (optional) restricts the reservation to a subset of
    slots; other slots get an all-zero mask.  When None, every slot is
    covered.
    """
    if n_slots <= 0:
        raise ValueError(f"n_slots must be positive, got {n_slots}")
    if start_sym < 0 or num_sym <= 0 or (start_sym + num_sym) > 14:
        raise ValueError(
            f"symbol range [{start_sym}, {start_sym + num_sym}) outside [0, 14]")
    bits = (((1 << num_sym) - 1) << start_sym) & 0x3FFF
    if target_slots is None:
        return [bits] * n_slots
    out = [0] * n_slots
    for s in target_slots:
        if not (0 <= s < n_slots):
            raise ValueError(f"target_slots entry {s} out of range [0, {n_slots})")
        out[s] = bits
    return out


def make_periodic_toggle_callback(period_s: float, n_slots: int,
                                  mask_when_on: int):
    """Build a sensing-policy callback that toggles on/off every ``period_s``.

    Designed to be passed to ``SpectrumSharingDApp.set_sensing_policy_logic``.
    Returns a closure with signature ``cb(now: float) -> tuple | None``:

      - On the FIRST call (``last_toggle is None``) the policy is
        activated with a uniform ``mask_when_on`` regardless of ``now``.
        Operator intuition: "active at startup, off ``period_s`` seconds
        later, then alternating."  ``last_toggle`` is stamped to ``now``
        so subsequent calls within ``period_s`` return None (no-op).
      - After ``period_s`` elapses, the callback flips state: if active,
        it sends ``deactivate=True`` with an all-zero mask; otherwise it
        re-activates with ``mask_when_on``.

    The closure stores its state in a dict (mutable across closure
    invocations).  Thread-safe to the extent that the callback is invoked
    serially from the indication worker.
    """
    if period_s <= 0:
        raise ValueError(f"period_s must be positive, got {period_s}")
    if n_slots <= 0:
        raise ValueError(f"n_slots must be positive, got {n_slots}")
    if mask_when_on < 0 or mask_when_on > 0x3FFF:
        raise ValueError(f"mask_when_on={mask_when_on:#x} not a valid 14-bit mask")

    # ``last_toggle`` starts as None so we explicitly recognise the
    # first call -- previously initialised to 0.0, which combined with
    # time.monotonic() (always returns a large positive in production)
    # made the first-tick behaviour look correct in practice but caused
    # the unit test to assert the opposite (using ``now=0.0``).  None
    # sentinel makes the intent explicit and tests can use any ``now``.
    state = {"last_toggle": None, "active": False}

    def _toggle_cb(now: float):
        if state["last_toggle"] is not None and now - state["last_toggle"] < period_s:
            return None
        state["last_toggle"] = now
        state["active"] = not state["active"]
        if state["active"]:
            return (True, [mask_when_on] * n_slots, False)
        return (True, [0] * n_slots, True)

    return _toggle_cb


class SpectrumSharingDApp(DApp):

    _SPECTRUM_JSON_BINARY_FIELDS = {
        "Spectrum-PRBBlacklistControl": ["blacklistedPRBs"],
        "Spectrum-PRBBlacklistReport": ["blacklistedPRBs"]
    }

    # dApp metadata
    DAPP_NAME = "SpectrumSharingDApp"
    DAPP_VERSION = "1.0.0"
    VENDOR = "WinesLab"
    E3AP_PROTOCOL_VERSION = "1.0.0"

    # IDs of interest for this dApp.
    #
    # Telemetry (IQ) comes from the OAI-L1-KPM SM at RF=2 — same shm
    # (/e3_ran_buffers) and a payload schema field-for-field equivalent
    # to what aerial dApps consume, so all dApps share one IQ pipeline.
    # On the ASN.1 channel the payload is APER-encoded L1KPM-Indication;
    # on the JSON channel it's an inline JSON object. We pick telemetry
    # id 1 ("iq_samples") from that SM's published list (e3_kpm_sm.c).
    #
    # Controls are split across two SMs:
    #  - PRB block      → SpectrumSM (RF=1, ctl=1, ASN.1 or JSON)
    #  - Sensing policy → SpectrumSM (RF=1, ctl=2, ASN.1 or JSON)
    # ``schedule_control`` calls take an explicit ranFunctionId so we
    # pick per call.
    RAN_FUNCTION_ID = 2
    TELEMETRY_ID = [1]
    CONTROL_ID = [1]

    # Spectrum SM (RF=1): sensing-range telemetry + PRB-block / sensing-policy control.
    PRB_CONTROL_RAN_FUNCTION_ID = 1
    SPECTRUM_TELEMETRY_ID_SENSING = 1  # Spectrum-SensingIndication stream
    SPECTRUM_CONTROL_ID_PRB_BLOCK = 1
    SPECTRUM_CONTROL_ID_SENSING_POLICY = 2

    # Symbol mask upper bound (14-bit bitmap).
    SENSING_SYMBOL_MASK_MAX = 0x3FFF

    # Slot duration at 30 kHz SCS (numerology 1). Used to derive the SHM
    # staleness bound from the ring depth reported in the shm header.
    _SLOT_DURATION_NS = 500_000
    # Fallback staleness bound until the FH ring header is read (see _shm_staleness_ns).
    _SHM_STALENESS_NS_DEFAULT = 50_000_000

    # PRBs on each side of the carrier centre (DC subcarrier) that exhibit
    # leakage artefacts and must be excluded from reports and control messages.
    # The band itself is derived from num_prbs in __init__ (DC sits at the
    # centre), so it tracks the active bandwidth instead of hardcoding 106 PRB.
    DC_LEAKAGE_GUARD_PRBS = 3

    # Defaults below assume: BW ≈ 40 MHz, center 3.6192 GHz, do_SRS=0 on
    # the gNB. FFT size is 2048 (or 1536 with USRP -E sampling). Noise-
    # floor threshold must be calibrated to the RU.

    def __init__(self, dapp_name: str = DAPP_NAME, dapp_version: str = DAPP_VERSION,
                 vendor: str = VENDOR, e3ap_protocol_version: str = E3AP_PROTOCOL_VERSION,
                 link: str = 'posix', transport: str = 'ipc',
                 detector: ThresholdDetector | None = None,
                 save_iqs: bool = False, control: bool = False,
                 center_freq: float = 3.6192e9, num_prbs: int = 106,
                 num_subcarrier_spacing: int = 30,
                 e_sampling: bool = False, encoding_method: str = "asn1",
                 sampling_threshold: int = 5,
                 max_samples_per_file: int = 46_080_000,
                 fp16_beta: float = 1.0 / 2048.0,
                 sensing_only: bool = True,
                 strict_sensing: bool = False,
                 min_sensing_symbols: int = 1, **kwargs):
        super().__init__(dapp_name=dapp_name, dapp_version=dapp_version, vendor=vendor,
                         e3ap_protocol_version=e3ap_protocol_version, link=link,
                         transport=transport, encoding_method=encoding_method, **kwargs)

        # Initialize spectrum encoder based on encoding method
        self._init_spectrum_encoder()

        # Custom control logic callbacks
        self._sampling_threshold_control_callback = None
        # Sensing-policy callback fires from the indication worker once per
        # _process_indication.  The callback decides when to emit a runtime
        # sensingPolicy control via spectrum_sm ctrl_id=2.  See
        # set_sensing_policy_logic() for the signature and the
        # make_periodic_toggle_callback helper for the common "toggle every
        # N seconds" use case.
        self._sensing_policy_callback = None

        # gNB radio configuration
        self.num_consecutive_subcarriers_for_prb: int = 12  # Fixed by LTE/NR standard
        self.num_prbs = num_prbs
        # DC-leakage guard band, centred on the carrier's DC subcarrier
        # (PRB num_prbs//2), clamped to the carrier. Derived from num_prbs so
        # e.g. a 273-PRB carrier strips PRBs around 136, not the hardcoded 50-55.
        _dc_center = self.num_prbs // 2
        self.DC_LEAKAGE_PRB_LOW = max(0, _dc_center - self.DC_LEAKAGE_GUARD_PRBS)
        self.DC_LEAKAGE_PRB_HIGH = min(self.num_prbs - 1, _dc_center + self.DC_LEAKAGE_GUARD_PRBS)
        self.num_subcarrier_spacing = num_subcarrier_spacing  # subcarrier spacing in kHz
        self.ofdm_symbol_size = num_prbs * self.num_consecutive_subcarriers_for_prb
        self.bw = (self.ofdm_symbol_size * self.num_subcarrier_spacing * 1e3)  # Hz
        self.center_freq = center_freq
        # cuBB /e3_ran_buffers presents subcarriers in natural PRB order
        # (PRB 0..N-1, sc 0..11) — no FFT-shift wrap, no zero-padded
        # extension beyond ofdm_symbol_size. So fft_size == ofdm_symbol_size
        # and the first-carrier offset is 0. e_sampling is preserved for
        # external knobs but no longer affects buffer sizing.
        self.fft_size = self.ofdm_symbol_size
        self.first_carrier_offset = 0
        if e_sampling:
            dapp_logger.info(
                "e_sampling=True ignored: /e3_ran_buffers layout is "
                "natural-order PRB×SC; 3/4 USRP-E sampling does not apply"
            )

        # dApp configuration
        self.save_iqs = save_iqs
        self.sampling_threshold = sampling_threshold
        # IQ capture files rotate every `max_samples_per_file` true IQ samples.
        self.max_samples_per_file = max_samples_per_file

        # Detection strategy
        if detector is None:
            raise ValueError("A ThresholdDetector instance must be provided via the 'detector' parameter")
        self._detector = detector
        dapp_logger.info(f"Detector: {type(self._detector).__name__}, threshold: {self._detector.threshold_db} dB")

        # Pre-allocated float32 I/Q/mag buffers reused every indication
        # (no complex128 intermediate). _abs_shifted_buf is the FFT-shifted
        # mag — pre-allocating avoids the np.roll allocation per frame.
        self._I_buf = np.empty(self.fft_size, dtype=np.float32)
        self._Q_buf = np.empty(self.fft_size, dtype=np.float32)
        self._mag_buf = np.empty(self.fft_size, dtype=np.float32)
        self._abs_shifted_buf = np.empty(self.fft_size, dtype=np.float32)

        # IQ recording
        if self.save_iqs:
            from iq_saver.iq_saver import IQSaver
            # Nominal capture rate of the on-disk IQ. Each indication carries one
            # OFDM symbol of fft_size post-FFT frequency-domain bins (OAI rxdataF);
            # fft_size * subcarrier_spacing is the ADC sample rate that produced
            # those bins, equivalently the total bandwidth the fft_size bins span.
            # This is core:sample_rate per the SigMF spec, NOT the ~100 Hz sensing
            # cadence at which indications arrive.
            sample_rate = self.fft_size * self.num_subcarrier_spacing * 1e3
            dapp_logger.info(f"Nominal IQ capture rate: {sample_rate / 1e6:.3f} Msps")
            # Detector decision window. The static detector averages `window` frames
            # before each PRB decision; the adaptive detector decides per frame. This
            # is recorded as dapp:average_over_frames and drives the annotation-time
            # compensation (see _corrected_annotation_start).
            if isinstance(self._detector, StaticThresholdDetector):
                average_over_frames = self._detector.window
            else:
                average_over_frames = 1
            # The dApp writes every received symbol undecimated, so the true
            # on-disk rate equals the nominal rate. effective_sample_rate would
            # only drop below core:sample_rate if the writer decimated on-device,
            # which it does not.
            effective_sample_rate = sample_rate
            self.iq_saver = IQSaver(
                base_path=LOG_DIR,
                center_freq=self.center_freq,
                bandwidth=self.bw,
                sample_rate=sample_rate,
                annotation_flush_interval=10,
                hw_info=f"FFT:{self.fft_size}, PRBs:{self.num_prbs}, E-sampling:{e_sampling}",
                description=(
                    f"5G NR Uplink capture from SpectrumSharing dApp"
                    f" - RAN Function {self.RAN_FUNCTION_ID}"
                    f" - detector: {type(self._detector).__name__}"
                    f" - threshold: {self._detector.threshold_db} dB"
                ),
                dtype="ci16_le",
                # domain="frequency": the captured samples are post-FFT
                # frequency-domain bins (OAI rxdataF), one OFDM symbol of fft_size
                # bins per record — not time-domain ADC samples. The explicit
                # marker is required because core:datatype is ci16_le, which the
                # SPEC §6.4 datatype policy would otherwise infer as time-domain.
                #
                # OFDM resource-grid geometry (spear-lake DAPP_SIGMF_FIELDS.yml,
                # §6.5) is emitted as component dims — a loader reconstructs the
                # occupied bandwidth as n_prbs * n_sc_per_prb. n_ants=1: the dApp
                # captures one antenna stream.
                #
                # dapp:layout is intentionally omitted: it describes the compact
                # resource-grid interleaving (as the aerial recorder emits), but
                # each on-disk record here is fft_size *padded* FFT bins for a
                # single symbol (fft_size != n_prbs * n_sc_per_prb), so that
                # reshape does not apply and would misdescribe the bytes.
                #
                # dapp:samples_per_slot, however, MUST be emitted explicitly:
                # omitting it is not neutral — the registry tells readers to
                # default it to n_ants*n_symbols*n_prbs*n_sc_per_prb, which would
                # slice these padded records at the wrong boundary. The truthful
                # on-disk slot stride is fft_size (padded) * n_symbols (n_ants=1).
                domain="frequency",
                n_prbs=self.num_prbs,
                n_sc_per_prb=self.num_consecutive_subcarriers_for_prb,
                n_ants=1,
                n_symbols=NUM_OFDM_SYMBOLS_PER_SLOT,
                samples_per_slot=self.fft_size * NUM_OFDM_SYMBOLS_PER_SLOT,
                subcarrier_spacing_khz=self.num_subcarrier_spacing,
                sampling_threshold=self.sampling_threshold,
                max_samples_per_file=self.max_samples_per_file,
                average_over_frames=average_over_frames,
                effective_sample_rate=effective_sample_rate,
                schema_version=DAPP_SIGMF_SCHEMA_VERSION,
            )

        self.control = control
        dapp_logger.info(f"Control is {'not ' if not self.control else ''}active")

        # PRB-block bookkeeping. The gNB install is REPLACE, not additive, so
        # whatever we send becomes the full gNB block set. Two independent
        # sources feed it — the detection loop and xApp control overrides — so
        # each change re-sends the reconciled union; a source that clears must
        # not wipe the other's PRBs. All three sets are mutated from both the
        # indication worker and operator/xApp threads, so guard with a lock.
        self._prb_block_lock = threading.Lock()
        self._prb_block_detect = set()  # contribution from the detection loop
        self._prb_block_xapp = set()    # contribution from xApp overrides
        self._prb_block_sent = set()    # what is currently installed on the gNB

        self.energyGui = kwargs.get('energyGui', False)
        self.iqPlotterGui = kwargs.get('iqPlotterGui', False)
        self.dashboard = kwargs.get('dashboard', False)
        if self.save_iqs:
            self._ground_truth_label = kwargs.get('ground_truth', "")
            self._ground_truth_lock = threading.Lock()

        # Thread-safe sample_idx for IQ saver annotation cross-reference
        self._sample_idx_lock = threading.Lock()
        self.sample_idx = None

        # /e3_ran_buffers reader (FP16 cuBB layout). Lazily opened on the
        # first indication so the gNB has time to create the shm region.
        self.fp16_beta = fp16_beta
        self._shm_reader = E3RanBuffersReader(fp16_beta=fp16_beta)
        # RF=1 sensing ranges arrive out-of-band via the /e3_l2_sensing ring;
        # cache the latest per (sfn, slot) for the RF=2 IQ handler to consume.
        self._sensing_reader = E3L2SensingReader()
        self._sensing_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._shm_open_failures = 0
        self._shm_indications_handled = 0
        self._shm_indications_dropped = 0
        self._shm_stale_dropped = 0
        self._shm_stats_log_interval = 1024
        # Sliding-window drop-rate heartbeat: when >50% of recent slots
        # are filter-dropped (dim or strict), the dashboard freezes
        # because no new rows are queued. Raise a WARN with the knobs to
        # fix it. Window = last 1000 slots, fires at most every 5000.
        self._drop_window: deque[int] = deque(maxlen=1000)
        self._drop_warn_last = 0

        self._ind_queue = queue.Queue(maxsize=4)
        self._ind_queue_dropped = 0
        self._ind_worker_stop = threading.Event()
        # Producer-stall instrumentation: track the worker's last
        # iteration timestamp + the max inter-iteration gap per
        # rolling-stats window. Spikes here cause downstream dashboard
        # freezes (5s producer pause = 5s of nothing in _pending_mag_rows
        # = dashboard freeze) — surfacing them in the stats log makes
        # the symptom traceable without attaching a debugger.
        self._ind_worker_last_iter_ns: int = 0
        self._ind_worker_max_gap_ns: int = 0
        self._ind_queue_dropped_last_log: int = 0

        # Per-stage latency deques (4096-sample rolling windows). e2e is the
        # gNB-stamp → recv interval; the rest are pipeline-stage durations.
        self._lat = {name: deque(maxlen=4096) for name in (
            "e2e", "decode", "meta", "shm", "qwait", "mag", "dash", "det", "work",
        )}
        self._lat_last_log_t = 0.0
        self._lat_log_interval_s = 5.0

        # Sensing ranges (the per-slot sensing-PUSCH time/frequency footprint)
        # arrive on the Spectrum SM (RF=1) as Spectrum-SensingIndication and are
        # cached by (sfn,slot) in _sensing_cache. The L1-KPM (RF=2) handler then
        # looks them up for the matching IQ slot — an explicit cross-SM
        # correlation. RF=1 also owns PRB-block / sensing-policy control; the
        # dApp subscribes to it so those controls are accepted.
        self.sensing_only = sensing_only
        # Strict mode: only display slots where the sensing window covers
        # EVERY symbol of the slot. Drops any slot where the MAC scheduler
        # granted any UE PUSCH (even one symbol), eliminating CP-overlap
        # and spectral-leakage bleed into a partial sensing window. The
        # cost is a sparser waterfall: under heavy UE traffic most slots
        # get dropped and the dashboard only shows quiet intervals.
        self.strict_sensing = strict_sensing
        if strict_sensing and not sensing_only:
            dapp_logger.warning(
                "strict_sensing requires sensing_only=True; ignoring strict mode")
            self.strict_sensing = False
        if self.strict_sensing:
            dapp_logger.info(
                "Sensing-window filter: STRICT — dashboard drops any slot whose "
                "sensing window doesn't cover the whole slot (i.e. any UE PUSCH "
                "grant present). Use --no-strict-sensing or remove the flag for "
                "the relaxed behaviour."
            )
        self._strict_dropped = 0
        # Minimum number of kept symbols required to emit the slot.
        # Below this threshold, the partial-sensing row would be too dim
        # (mean of <N symbols) compared to full-sensing rows (mean of 14)
        # and shows up as a visible gap in the waterfall. Default 2 drops
        # the typical "only sym 13 survived = CP bleed" cases without
        # going to full strict mode.
        self.min_sensing_symbols = max(1, int(min_sensing_symbols))
        self._dim_dropped = 0
        if self.min_sensing_symbols > 1 and self.sensing_only:
            dapp_logger.info(
                "Sensing-window filter: relaxed (drop slots with < %d "
                "kept symbols to even out waterfall brightness)",
                self.min_sensing_symbols,
            )
        # Sensing-coverage bookkeeping (how many L1 slots carried a non-empty
        # sensing window), logged periodically and at shutdown.
        self._sensing_slots_total = 0
        self._sensing_slots_with_ranges = 0
        self._sensing_log_last = 0
        self._SENSING_LOG_INTERVAL = 1024
        dapp_logger.info(
            f"Sensing-window filter: {'enabled' if self.sensing_only else 'disabled'} "
            f"(sensing ranges arrive on the Spectrum SM RF={self.PRB_CONTROL_RAN_FUNCTION_ID}, "
            f"are cached by (sfn,slot), and correlated to the matching L1-KPM "
            f"RF={self.RAN_FUNCTION_ID} IQ slot; dashboard rows are masked to sensing-PUSCH "
            f"symbols when enabled)"
        )

        self._detector_run_interval = 4
        self._detector_run_counter = 0
        self._ind_worker_thread = threading.Thread(
            target=self._indication_worker,
            name="dapp_ind_worker",
            daemon=True,
        )
        self._ind_worker_thread.start()
        # Wall-clock diagnostic thread.  Wakes once per second and checks
        # whether shm indications are still flowing — the existing count-
        # driven heartbeat in _count_handled cannot fire if the inbound
        # thread itself has stalled (gNB crashed, libe3 dead, etc.).
        # Decoupled from any data plane so it always runs.
        self._diag_thread_stop = threading.Event()
        self._diag_last_handled = 0
        self._diag_silent_ticks = 0
        self._diag_thread = threading.Thread(
            target=self._diagnostic_loop,
            name="dapp_diag",
            daemon=True,
        )
        self._diag_thread.start()

        if self.energyGui:
            from visualization.energy import EnergyPlotter
            self.sig_queue = queue.Queue()
            self.energyPlotter = EnergyPlotter(
                self.fft_size, bw=self.bw, center_freq=self.center_freq
            )

        if self.iqPlotterGui:
            from visualization.iq import IQPlotter
            self.iq_queue = queue.Queue()
            self.iqPlotter = IQPlotter(
                buffer_size=500, fft_size=self.fft_size,
                bw=self.bw, center_freq=self.center_freq,
            )

        # Latest detector output, overlaid on every published waterfall frame.
        # Detection runs every _detector_run_interval slots but the waterfall
        # publishes every slot, so we cache the mask/threshold between runs.
        self._last_det_mask = None
        self._last_det_thr = None
        if self.dashboard:
            from visualization.subcarrier_pub import SubcarrierPublisher

            self.demo = SubcarrierPublisher(
                zmq_port=int(kwargs.get('viz_zmq_port', 5559)),
                web_port=int(kwargs.get('viz_web_port', 5001)),
                num_prbs=self.num_prbs,
                spawn_viz=not kwargs.get('external_viz', False),
            )

    def _init_spectrum_encoder(self):
        """Initialize the spectrum encoder based on the encoding method"""
        match self.encoding_method:
            case "asn1":
                asn_file_path = os.path.join(os.path.dirname(__file__), "defs", "e3sm_spectrum.asn")
                self.spectrum_encoder = asn1tools.compile_files(asn_file_path, codec="per")
            case "json":
                json_schema_path = os.path.join(os.path.dirname(__file__), "defs", "e3sm_spectrum.json")
                with open(json_schema_path, 'r') as f:
                    self.spectrum_schema = json.load(f)
                self.spectrum_validator_cls = jsonschema.validators.validator_for(self.spectrum_schema)
                self.spectrum_validator_cls.check_schema(self.spectrum_schema)
                self.spectrum_registry = self._build_spectrum_registry()
                self.spectrum_encoder = "json"
                dapp_logger.info("Spectrum JSON encoder initialized")
            case _:
                raise ValueError(f"Unsupported encoding method: {self.encoding_method}")

    def _build_spectrum_registry(self):
        import referencing
        import referencing.jsonschema
        resource = referencing.Resource.from_contents(
            self.spectrum_schema,
            default_specification=referencing.jsonschema.DRAFT202012,
        )
        base_uri = self.spectrum_schema.get("$id", "urn:e3sm-spectrum-schema")
        return referencing.Registry().with_resource(base_uri, resource)

    def _validate_spectrum_message(self, message_type: str, data: dict) -> None:
        """Validate a spectrum message dict against its schema definition.

        Same gotcha as e3interface.e3_encoder: passing the subschema directly
        breaks `#/$defs/...` resolution because the validator's current
        resource is the subschema (no $defs). Use a $ref into the registered
        URI so the registry handles ref resolution against the full schema.
        """
        if message_type not in self.spectrum_schema.get("$defs", {}):
            return
        base_uri = self.spectrum_schema.get("$id", "urn:e3sm-spectrum-schema")
        validator = self.spectrum_validator_cls(
            {"$ref": f"{base_uri}#/$defs/{message_type}"},
            registry=self.spectrum_registry,
        )
        validator.validate(data)

    def set_sampling_threshold_control_logic(self, callback):
        """Set a custom control logic callback.

        The callback is invoked after each detection with the signature::

            callback(prb_blacklist: np.ndarray, power_db: np.ndarray) -> tuple[bool, int]

        Args:
            callback: A callable that takes:
                - prb_blacklist (np.ndarray): Blacklisted PRB indices.
                - power_db (np.ndarray): Per-bin signal power in dB, shape
                  ``(fft_size,)``, first-carrier-aligned.  Available in both
                  static and adaptive mode.

                Returns:
                - update_sampling (bool): Whether to update the sampling threshold.
                - sampling_threshold (int): New sampling threshold value (0–100).

        Example::

            def my_logic(prb_blacklist, power_db):
                update = len(prb_blacklist) > 10
                return update, 10 if update else 5

            dapp.set_sampling_threshold_control_logic(my_logic)
        """
        if callback is not None and not callable(callback):
            raise ValueError("Callback must be callable")
        self._sampling_threshold_control_callback = callback
        dapp_logger.info(f"Custom control logic callback {'set' if callback else 'removed'}")

    def set_sensing_policy_logic(self, callback):
        """Install a per-indication callback that decides when to emit a
        runtime sensingPolicy control to the gNB.

        The callback is invoked at the end of every indication-worker
        iteration with signature::

            callback(now: float) -> tuple[bool, list[int], bool] | None

        where ``now`` is ``time.monotonic()``.  The return value is either
        ``None`` (do nothing this tick) or a tuple ``(send_now,
        mask_per_slot, deactivate)``:

          - ``send_now``: if False the rest of the tuple is ignored.
          - ``mask_per_slot``: per-slot 14-bit bitmap.  Length MUST equal
            the gNB's ``numb_slots_frame`` (e.g. 20 at mu=1); a mismatch
            yields a negative ACK on the gNB and an error log here.
          - ``deactivate``: when True the gNB clears the policy regardless
            of mask contents.  Pair with an empty mask when toggling off.

        The callback handles its own cadence (e.g. "fire once every 10
        seconds") — the harness just polls it at indication rate, which
        is cheap (one float comparison) and never blocks.

        See ``make_periodic_toggle_callback`` for the common
        "toggle on/off every N seconds" use case.
        """
        if callback is not None and not callable(callback):
            raise ValueError("Callback must be callable")
        self._sensing_policy_callback = callback
        dapp_logger.info(
            f"Sensing-policy callback {'set' if callback else 'removed'}"
        )

    def set_ground_truth_label(self, label: str):
        with self._ground_truth_lock:
            self._ground_truth_label = label
        dapp_logger.info(f"Ground truth label set to: {label!r}")

    def create_prb_block_control(self, blocked_prbs: list[int],
                                 update_sampling: bool = False,
                                 validity_period: int = None) -> bytes:
        """Create a PRB-block control message, wrapped in the
        Spectrum-DAppControlData envelope.

        Drives the gNB's per-MAC prb_block_state_t via set_prb_block_mask():
        each PRB in ``blocked_prbs`` gets every UL symbol marked occupied,
        so downstream scheduling treats it as unavailable.

        Args:
            blocked_prbs: List of PRB indices to block on the UL
            update_sampling: Whether to include the updated sampling threshold
            validity_period: How long this block is valid in seconds (optional)

        Returns:
            Encoded bytes for E3-DAppControlAction.actionData
        """
        control_data = {"blacklistedPRBs": blocked_prbs}
        if update_sampling:
            control_data["samplingThreshold"] = self.sampling_threshold
        if validity_period is not None:
            control_data["validityPeriod"] = validity_period
        dapp_logger.debug(control_data)
        return self._encode_dapp_control_envelope(
            type_value="prbBlacklist",
            payload_key="prbBlacklistControl",
            inner_type="Spectrum-PRBBlacklistControl",
            inner=control_data,
        )

    def _reconcile_prb_blocks(self, *, detect: set | None = None,
                              xapp: set | None = None,
                              update_sampling: bool = False) -> None:
        """Recompute the union of all PRB-block sources and, if it changed,
        push the FULL set to the gNB (install is REPLACE, not additive).

        Thread-safe: the detection worker (``detect=``) and operator/xApp
        threads (``xapp=``) both call this. Sending only a delta would let one
        source's change unblock PRBs the other still wants blocked, so we always
        re-send the reconciled union.
        """
        with self._prb_block_lock:
            if detect is not None:
                self._prb_block_detect = set(detect)
            if xapp is not None:
                self._prb_block_xapp = set(xapp)
            desired = self._prb_block_detect | self._prb_block_xapp
            if desired == self._prb_block_sent:
                return
            blocked = sorted(desired)
            control_payload = self.create_prb_block_control(
                blocked_prbs=blocked, update_sampling=update_sampling
            )
            self.e3_interface.schedule_control(
                dappId=self.dapp_id,
                ranFunctionId=self.PRB_CONTROL_RAN_FUNCTION_ID,
                controlId=self.SPECTRUM_CONTROL_ID_PRB_BLOCK,
                actionData=control_payload,
            )
            self._prb_block_sent = set(desired)
        dapp_logger.info(
            "PRB block set updated: %d PRB(s) installed (detect=%d, xApp=%d)",
            len(blocked), len(self._prb_block_detect), len(self._prb_block_xapp),
        )

    def clear_prb_blocks(self) -> bool:
        """Unconditionally clear ALL PRB blocks on the gNB.

        Sends an empty blacklistedPRBs list (the gNB treats this as a full
        UL+DL clear via set_prb_block_mask(NULL)) and resets every source set so
        subsequent detections re-block from scratch. Sent unconditionally — even
        when this instance's own ``_prb_block_sent`` is empty — because a prior
        dApp instance may have died with blocks still installed on the sticky
        gNB. Called once at startup (see send_subscription_request) and available
        as an operator action. Returns True if the clear was queued.
        """
        if not getattr(self, "dapp_id", None):
            dapp_logger.warning("clear_prb_blocks: dApp not connected; dropping")
            return False
        with self._prb_block_lock:
            n = len(self._prb_block_sent)
            self._prb_block_detect.clear()
            self._prb_block_xapp.clear()
            control_payload = self.create_prb_block_control(blocked_prbs=[])
            self.e3_interface.schedule_control(
                dappId=self.dapp_id,
                ranFunctionId=self.PRB_CONTROL_RAN_FUNCTION_ID,
                controlId=self.SPECTRUM_CONTROL_ID_PRB_BLOCK,
                actionData=control_payload,
            )
            self._prb_block_sent.clear()
        dapp_logger.info(f"clear_prb_blocks: explicit UL+DL clear sent (was {n} PRB(s))")
        return True

    def create_sensing_policy_control(self, mask_per_slot: list[int],
                                      deactivate: bool = False,
                                      validity_period: int | None = None) -> bytes:
        """Build the Spectrum-DAppControlData{sensingPolicy} envelope bytes
        carried by E3-DAppControlAction.actionData.

        Validates mask_per_slot in-Python (length > 0, each value in
        0..0x3FFF) so encoder errors surface as ValueError instead of
        asn1tools tracebacks.
        """
        if not isinstance(mask_per_slot, (list, tuple)) or len(mask_per_slot) == 0:
            raise ValueError("mask_per_slot must be a non-empty list[int]")
        for i, m in enumerate(mask_per_slot):
            if not isinstance(m, int) or m < 0 or m > self.SENSING_SYMBOL_MASK_MAX:
                raise ValueError(
                    f"mask_per_slot[{i}]={m!r} is not a valid 14-bit symbol mask "
                    f"(0..{self.SENSING_SYMBOL_MASK_MAX:#06x})"
                )
        inner = {"maskPerSlot": list(mask_per_slot), "deactivate": bool(deactivate)}
        if validity_period is not None:
            inner["validityPeriod"] = int(validity_period)
        dapp_logger.debug(f"sensingPolicy control payload: {inner}")
        return self._encode_dapp_control_envelope(
            type_value="sensingPolicy",
            payload_key="sensingPolicyControl",
            inner_type="Spectrum-SensingPolicyControl",
            inner=inner,
        )

    def send_sensing_policy(self, mask_per_slot: list[int], *,
                            deactivate: bool = False,
                            validity_period: int | None = None) -> bool:
        """Queue a runtime sensingPolicy control to the gNB (RF=1, ctrl_id=2).

        Thread-safe (the underlying E3Interface.schedule_control is queued
        via queue.Queue).  Returns True if the control was queued; False
        on encoder error or not-connected.  Note: the ACK from the gNB
        arrives asynchronously on the inbound thread and is logged at
        INFO/DEBUG -- this method does NOT wait for it.
        """
        if not getattr(self, "dapp_id", None):
            dapp_logger.warning("send_sensing_policy: dApp not connected; dropping")
            return False
        try:
            payload = self.create_sensing_policy_control(
                mask_per_slot=mask_per_slot,
                deactivate=deactivate,
                validity_period=validity_period,
            )
        except Exception as exc:
            dapp_logger.error(f"send_sensing_policy: encode failed: {exc}")
            return False
        self.e3_interface.schedule_control(
            dappId=self.dapp_id,
            ranFunctionId=self.PRB_CONTROL_RAN_FUNCTION_ID,  # SpectrumSM (RF=1)
            controlId=self.SPECTRUM_CONTROL_ID_SENSING_POLICY,
            actionData=payload,
        )
        dapp_logger.info(
            "Queued sensingPolicy: n_slots=%d deactivate=%s validity=%s",
            len(mask_per_slot), deactivate, validity_period,
        )
        return True

    def create_prb_blacklist_report(self, blacklisted_prbs: list[int]) -> bytes:
        """Create a PRB blacklist report message, wrapped in the
        Spectrum-DAppReportData envelope.

        Args:
            blacklisted_prbs: List of PRB indices

        Returns:
            Encoded bytes for E3-DAppReport.reportData
        """
        report_data = {"blacklistedPRBs": blacklisted_prbs}
        dapp_logger.debug(report_data)
        return self._encode_dapp_report_envelope(
            type_value="prbBlacklist",
            payload_key="prbBlacklistReport",
            inner_type="Spectrum-PRBBlacklistReport",
            inner=report_data,
        )

    # ---- Envelope encode/decode helpers ----------------------------- #
    # All Spectrum-* envelopes share the same (Type, Payload-CHOICE) shape.
    # _encode_envelope / _decode_envelope parametrise that shape; the four
    # flavours (control/report encode, indication/xapp-control decode) are
    # one-line wrappers below.

    def _encode_envelope(self, *, msg_type: str, type_field: str, payload_field: str,
                         type_value: str, payload_key: str, inner_type: str,
                         inner: dict) -> bytes:
        if self.encoding_method == "asn1":
            # The gNB's ASN.1 "*Data" envelopes are payload-only SEQUENCEs: the
            # CHOICE alternative is the on-wire discriminator, so NO separate
            # type field is encoded (see src/spectrum/defs/e3sm_spectrum.asn,
            # aligned byte-for-byte with the merged develop-dapp gNB).
            return self.spectrum_encoder.encode(msg_type, {
                payload_field: (payload_key, inner),
            })
        if self.encoding_method == "json":
            # JSON keeps an explicit, human-readable type key next to the
            # payload. The gNB's JSON decoder treats it as OPTIONAL (and only
            # validates it when present), so emitting it is safe and aids
            # cross-checking / dashboards.
            prepared = JsonE3Encoder.prepare_data_for_json_encode(
                inner_type, inner.copy(), self._SPECTRUM_JSON_BINARY_FIELDS
            )
            envelope = {type_field: type_value, payload_field: {payload_key: prepared}}
            self._validate_spectrum_message(msg_type, envelope)
            return json.dumps(envelope).encode("utf-8")
        raise ValueError(f"Unsupported encoding method: {self.encoding_method}")

    def _decode_envelope(self, data: bytes, *, msg_type: str, type_field: str,
                         payload_field: str, inner_type_map: dict,
                         type_by_key: dict | None = None) -> dict:
        """Returns ``{"type": str | None, "payload_key": str, "payload": dict}``.

        ``type_by_key`` maps a CHOICE payload key to its human-readable type
        string; it is used on the ASN.1 path where the type is not carried on
        the wire (derived from the CHOICE alternative instead). On the JSON
        path the type comes from the envelope's optional ``type_field``.
        """
        type_by_key = type_by_key or {}
        if self.encoding_method == "asn1":
            env = self.spectrum_encoder.decode(msg_type, data)
            payload_key, payload = env[payload_field]
            return {"type": type_by_key.get(payload_key),
                    "payload_key": payload_key, "payload": payload}
        if self.encoding_method == "json":
            env = json.loads(data.decode("utf-8"))
            self._validate_spectrum_message(msg_type, env)
            payload_key, payload = next(iter(env[payload_field].items()))
            inner_type = inner_type_map.get(payload_key)
            if inner_type is not None:
                payload = JsonE3Encoder.prepare_data_from_json_decode(
                    inner_type, payload, self._SPECTRUM_JSON_BINARY_FIELDS
                )
            return {"type": env.get(type_field, type_by_key.get(payload_key)),
                    "payload_key": payload_key, "payload": payload}
        raise ValueError(f"Unsupported encoding method: {self.encoding_method}")

    def _encode_dapp_control_envelope(self, *, type_value, payload_key, inner_type, inner) -> bytes:
        return self._encode_envelope(
            msg_type="Spectrum-DAppControlData",
            type_field="controlType", payload_field="controlPayload",
            type_value=type_value, payload_key=payload_key,
            inner_type=inner_type, inner=inner,
        )

    def _encode_dapp_report_envelope(self, *, type_value, payload_key, inner_type, inner) -> bytes:
        return self._encode_envelope(
            msg_type="Spectrum-DAppReportData",
            type_field="reportType", payload_field="reportPayload",
            type_value=type_value, payload_key=payload_key,
            inner_type=inner_type, inner=inner,
        )

    def _decode_xapp_control_envelope(self, data: bytes) -> dict:
        return self._decode_envelope(
            data,
            msg_type="Spectrum-XAppControlData",
            type_field="controlType", payload_field="controlPayload",
            inner_type_map={
                "prbBlockedControl": "Spectrum-PRBBlockedControl",
                "configControl": "Spectrum-ConfigControl",
            },
            type_by_key={"prbBlockedControl": "prbBlocked", "configControl": "config"},
        )

    def _encode_spectrum_message(self, message_type: str, data: dict) -> bytes:
        """Encode a spectrum message using the configured encoding method."""
        if self.encoding_method == "asn1":
            return self.spectrum_encoder.encode(message_type, data)
        if self.encoding_method == "json":
            json_data = JsonE3Encoder.prepare_data_for_json_encode(
                message_type, data.copy(), self._SPECTRUM_JSON_BINARY_FIELDS
            )
            self._validate_spectrum_message(message_type, json_data)
            return json.dumps(json_data).encode("utf-8")
        raise ValueError(f"Unsupported encoding method: {self.encoding_method}")

    def decode_config_control(self, data: bytes) -> dict:
        """Decode a spectrum xApp control message and return the
        Spectrum-ConfigControl payload.

        Accepts bytes from E3-XAppControlAction.xAppControlData. Raises
        ValueError if the envelope holds a different variant.
        """
        env = self._decode_xapp_control_envelope(data)
        if env["type"] != "config" or env["payload_key"] != "configControl":
            raise ValueError(
                f"Expected configControl envelope, got type={env['type']!r} "
                f"payload_key={env['payload_key']!r}"
            )
        return env["payload"]

    @override
    def _decode_ran_function_data(self, data_bytes: bytes) -> dict | None:
        """Decode the ``ranFunctionData`` attached to a SetupResponse entry."""
        return self._decode_spectrum_message("Spectrum-RanFunctionData", data_bytes)

    @override
    def send_subscription_request(self, subscriptionTime=None, periodicity=None) -> bool:
        """Subscribe to the L1-KPM SM (RF=2, IQ) and the Spectrum SM (RF=1).

        RF=2 carries the IQ shm pointer + validSymbolMask. RF=1 carries the
        sensing-range telemetry (Spectrum-SensingIndication → /e3_l2_sensing)
        and accepts the PRB-block / sensing-policy controls. The RF=1
        subscription therefore requests the sensing telemetry id in addition
        to the two control ids.

        Returns the L1 subscription's scheduled status; the Spectrum-SM
        subscription is best-effort.

        The two requests are queued ~50 ms apart instead of back-to-back.
        With zero spacing libe3's setup-loop on the gNB only sees one of
        the two requests (likely a slow-joiner / batching artefact on
        the ZMQ subscriber side — the dApp's outbound thread enqueues
        and sends both within ~1 ms which is below the SUB socket's
        per-message handling window). 50 ms is well above the threshold
        and an imperceptible delay at dApp startup."""
        l1_scheduled = super().send_subscription_request(subscriptionTime, periodicity)

        # Stagger the second subscription so libe3's setup loop processes
        # the first one cleanly before the second arrives. Without this,
        # the RF=2 request gets dropped silently and L1-KPM never starts,
        # which in turn means /e3_ran_buffers is never created (the PHY
        # tap is gated on e3_kpm_sm_has_subscribers()).
        time.sleep(0.05)

        # The Spectrum SM (RF=1) is not optional: without it _sensing_cache
        # stays empty forever (every slot then detects unmasked) and the
        # PRB-block / sensing-policy controls are silently dropped gNB-side
        # because they target an unsubscribed function id. Retry a few times
        # against the slow-joiner window, then fail the whole subscription.
        spectrum_scheduled = False
        for attempt in range(3):
            spectrum_scheduled = self.e3_interface.send_subscription_request(
                self.dapp_id,
                self.PRB_CONTROL_RAN_FUNCTION_ID,  # Spectrum SM (RF=1)
                [self.SPECTRUM_TELEMETRY_ID_SENSING],  # sensing-range telemetry
                [self.SPECTRUM_CONTROL_ID_PRB_BLOCK, self.SPECTRUM_CONTROL_ID_SENSING_POLICY],
                subscriptionTime,
                periodicity,
            )
            if spectrum_scheduled:
                break
            dapp_logger.warning(
                f"Spectrum SM RF={self.PRB_CONTROL_RAN_FUNCTION_ID} subscription not "
                f"scheduled (attempt {attempt + 1}/3); retrying"
            )
            time.sleep(0.05)
        if not spectrum_scheduled:
            dapp_logger.error(
                f"Spectrum SM RF={self.PRB_CONTROL_RAN_FUNCTION_ID} subscription failed; "
                "sensing telemetry and PRB/sensing-policy controls would be unavailable"
            )
            return False
        if self.control:
            # Drop any sticky PRB blocks a previous dApp instance left installed
            # on the gNB before this instance starts blocking from a clean slate.
            self.clear_prb_blocks()
        return l1_scheduled

    def _decode_spectrum_message(self, message_type: str, data: bytes) -> dict:
        """Decode a spectrum message using the configured encoding method."""
        if self.encoding_method == "asn1":
            return self.spectrum_encoder.decode(message_type, data)
        if self.encoding_method == "json":
            decoded_data = json.loads(data.decode("utf-8"))
            self._validate_spectrum_message(message_type, decoded_data)
            return JsonE3Encoder.prepare_data_from_json_decode(
                message_type, decoded_data, self._SPECTRUM_JSON_BINARY_FIELDS
            )
        raise ValueError(f"Unsupported encoding method: {self.encoding_method}")

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _corrected_annotation_start(sample_idx: int, window: int,
                                    samples_per_indication: int) -> int:
        """IQ-sample index of the first frame contributing to a decision.

        ``sample_idx`` is the start index of the most recent indication (the
        last frame of the averaging window). A detector that averages ``window``
        frames before deciding bases that decision on the preceding ``window``
        indications, each carrying ``samples_per_indication`` true IQ samples, so
        the annotation must point back to the first of them. ``window == 1`` (no
        averaging) leaves the index unchanged. The result is clamped at 0.
        """
        return max(0, sample_idx - (window - 1) * samples_per_indication)

    def _compute_prb_blacklist(self, detected_prbs: np.ndarray) -> np.ndarray:
        """Drop only the guard band (>=num_prbs).  Low-PRB filtering used
        to mask off PRACH/PUCCH territory but the gNB now handles those
        collisions gracefully (see gNB_scheduler_srs.c / gNB_scheduler_RA.c
        post-fef0e18d43), so the dApp ships every detected PRB inside the
        BWP and lets the scheduler decide.  DC-leakage stripping happens
        upstream in the detector."""
        return detected_prbs[detected_prbs < self.num_prbs]

    def _build_annotation_fields(
        self,
        all_prbs: np.ndarray,
        prb_blk_list: np.ndarray,
        power_db: np.ndarray,
        noise_floor_db: np.ndarray | None,
        timestamp,
        comment: str,
        control_fired: bool,
    ) -> dict:
        """Build the full annotation metadata dict for an IQSaver annotation.

        Returns a dict suitable for keyword-unpacking into ``add_annotation``.
        Per-PRB summaries are included so individual recordings are fully
        self-documenting without requiring re-analysis.

        all_prbs: every PRB above the detection threshold (unfiltered, including
                  protected-zone PRBs).  prb_blk_list: the filtered blacklist
                  (eligible zone only, empty when control is off).
        """
        sc = self.num_consecutive_subcarriers_for_prb

        # All detected peaks (unfiltered) — always recorded for analysis
        all_sc_starts = all_prbs * sc
        all_power_per_prb = [float(power_db[s: s + sc].mean()) for s in all_sc_starts]
        if isinstance(self._detector, StaticThresholdDetector):
            all_thresh_per_prb = [self._detector.threshold_db] * len(all_prbs)
        elif noise_floor_db is not None:
            all_thresh_per_prb = [float(noise_floor_db[s: s + sc].mean()) for s in all_sc_starts]
        else:
            all_thresh_per_prb = [self._detector.threshold_db] * len(all_prbs)

        # Per-PRB record of the subcarriers that actually crossed the
        # threshold in the current frame, with their dB magnitudes. Derived
        # from the instantaneous power-vs-threshold comparison, not from the
        # detector's blocked mask: under the adaptive detector that mask is
        # embargo-extended and can stay true after the instantaneous SNR has
        # fallen back below threshold. PRB indices are stored as strings
        # because this structure is serialised to SigMF JSON, where object
        # keys are always strings; each value is a list of
        # [sc_index_within_prb, db] pairs, where sc_index_within_prb is the
        # local 0..(sc-1) offset inside the PRB (not the absolute FFT-bin index).
        if isinstance(self._detector, StaticThresholdDetector) or noise_floor_db is None:
            detected_now = power_db > self._detector.threshold_db
        else:
            detected_now = power_db >= noise_floor_db + self._detector.threshold_db
        above_threshold_sc_db_per_prb: dict[str, list[list[float]]] = {}
        for prb, s in zip(all_prbs, all_sc_starts):
            local_offsets = np.where(detected_now[s: s + sc])[0].tolist()
            above_threshold_sc_db_per_prb[str(int(prb))] = [
                [j, float(power_db[s + j])] for j in local_offsets
            ]

        label = "prb_control" if control_fired else "prb_detection"

        base = dict(
            label=label,
            timestamp=timestamp,
            comment=comment,
            all_detected_prbs=all_prbs.tolist(),
            all_power_db_per_prb=all_power_per_prb,
            above_threshold_sc_db_per_prb=above_threshold_sc_db_per_prb,
            all_threshold_db_per_prb=all_thresh_per_prb,
            # prb_blacklist only populated when a control message was actually sent;
            # otherwise the eligible-zone filtering is irrelevant to the recording.
            prb_blacklist=prb_blk_list.tolist() if control_fired else [],
            noise_threshold=self._detector.threshold_db,
            power_db_max=float(power_db.max()),
            control_action="blacklist" if control_fired else "none",
        )

        if isinstance(self._detector, StaticThresholdDetector):
            base.update(
                detector="static",
                window_frames=self._detector.window,
            )
        elif isinstance(self._detector, AdaptiveThresholdDetector) and noise_floor_db is not None:
            blk_sc_starts = prb_blk_list * sc
            nf_per_prb = [float(noise_floor_db[s: s + sc].mean()) for s in blk_sc_starts]
            snr_per_prb = [float(power_db[s: s + sc].mean() - noise_floor_db[s: s + sc].mean())
                           for s in blk_sc_starts]
            base.update(
                detector="adaptive",
                hist_depth=self._detector.hist_depth,
                embargo_timeout_secs=self._detector.embargo_timeout_secs,
                noise_floor_per_prb=nf_per_prb,
                snr_db_per_prb=snr_per_prb,
            )

        return base

    def _ensure_shm_open(self) -> bool:
        """Lazy-open /e3_ran_buffers. The gNB creates it on first UL slot,
        so opening at dApp startup races with the gNB. Returns True if
        the reader is ready to serve reads."""
        if self._shm_reader.is_open:
            return True
        try:
            self._shm_reader.open()
            dapp_logger.info(
                f"[SHM] /e3_ran_buffers opened (num_fh_rows="
                f"{self._shm_reader.num_fh_rows}, fp16_beta={self.fp16_beta:g})"
            )
            return True
        except FileNotFoundError:
            self._shm_open_failures += 1
            if self._shm_open_failures == 1 or (self._shm_open_failures % 100) == 0:
                dapp_logger.info(
                    f"[SHM] /e3_ran_buffers not present yet "
                    f"(open attempts={self._shm_open_failures}); waiting for gNB"
                )
            return False
        except Exception as exc:
            self._shm_open_failures += 1
            dapp_logger.error(f"[SHM] /e3_ran_buffers open failed: {exc}")
            return False

    def _count_read_drop(self) -> None:
        """Shm-open or row-read failure (e.g. out-of-range write_index)."""
        self._shm_indications_dropped += 1
        if self._shm_indications_dropped == 1:
            dapp_logger.info(
                "[SHM] first read drop (shm not yet open or row OOR); "
                "subsequent drops at debug level"
            )
        else:
            dapp_logger.debug(
                f"[SHM] read drop (total={self._shm_indications_dropped})"
            )

    def _count_handled(self, p: SlotPointer, iq_symbols) -> None:
        """Bump the handled counter and emit periodic [SHM] stats."""
        self._shm_indications_handled += 1
        if self._shm_indications_handled == 1:
            dapp_logger.info(
                f"[SHM] first slot pulled: sfn={p.sfn} slot={p.slot} "
                f"buf={p.fh_buffer_index} row={p.fh_write_index} "
                f"symbols={len(iq_symbols)} samples_per_symbol={iq_symbols[0].size}"
            )
        elif (self._shm_indications_handled % self._shm_stats_log_interval) == 0:
            total_sent = self._shm_indications_handled + self._shm_indications_dropped
            read_drop_pct = 100.0 * self._shm_indications_dropped / max(1, total_sent)
            dapp_logger.info(
                f"[SHM] stats: handled={self._shm_indications_handled} "
                f"read_drops={self._shm_indications_dropped} ({read_drop_pct:.2f}%) "
                f"dim_dropped={self._dim_dropped} strict_dropped={self._strict_dropped}"
            )
            # Worker-stall watch.  Surface max-iter-gap + per-window
            # queue drops.  A spike here (gap >> 1ms typical) is the
            # smoking gun for an upstream stall that would show up
            # downstream as a dashboard freeze.  Bumped to WARN when
            # the gap exceeds 1 second so the operator notices.
            worker_drops_window = (
                self._ind_queue_dropped - self._ind_queue_dropped_last_log
            )
            self._ind_queue_dropped_last_log = self._ind_queue_dropped
            max_gap_ms = self._ind_worker_max_gap_ns / 1e6
            self._ind_worker_max_gap_ns = 0  # reset per window
            log_fn = dapp_logger.warning if max_gap_ms > 1000 else dapp_logger.info
            log_fn(
                "[WORKER] indication worker: max_iter_gap_ms=%.1f "
                "queue_drops_window=%d queue_drops_total=%d qsize=%d",
                max_gap_ms, worker_drops_window,
                self._ind_queue_dropped, self._ind_queue.qsize(),
            )
        # Sliding-window drop-rate heartbeat. Hint the operator how to
        # un-freeze the dashboard instead of leaving them staring at it.
        if (len(self._drop_window) >= 200
                and (self._shm_indications_handled - self._drop_warn_last) >= 5000):
            drop_rate = sum(self._drop_window) / len(self._drop_window)
            if drop_rate > 0.5:
                self._drop_warn_last = self._shm_indications_handled
                dapp_logger.warning(
                    "WARN: dashboard filter-drop rate >50%% — heavy UE UL is "
                    "collapsing sensing windows. Lower --min-sensing-symbols "
                    "(current=%d) or pass --no-sensing-only. "
                    "(window=%d slots, rate=%.1f%%)",
                    self.min_sensing_symbols,
                    len(self._drop_window), 100.0 * drop_rate,
                )

    # ---- Sensing-coverage bookkeeping ------------------------------- #
    # Tally how many L1-KPM (RF=2) slots found a cached RF=1 sensing window
    # (via the (sfn,slot) cross-SM correlation) for ops visibility — a low
    # ratio flags lost/late RF=1 indications.

    def _tally_sensing(self, has_ranges: bool) -> None:
        self._sensing_slots_total += 1
        if has_ranges:
            self._sensing_slots_with_ranges += 1
        total = self._sensing_slots_total
        if (total - self._sensing_log_last) >= self._SENSING_LOG_INTERVAL:
            self._sensing_log_last = total
            pct = 100.0 * self._sensing_slots_with_ranges / total
            dapp_logger.info(
                f"[SENSING] L1_slots={total} with_ranges={self._sensing_slots_with_ranges} "
                f"({pct:.1f}%)"
            )

    @staticmethod
    def _build_keep_sym_mask(ranges) -> int:
        """OR each sensing range's symbol span into a 14-bit keep mask
        (bit i set ⇒ symbol i carries sensing PUSCH). Out-of-range bits
        are masked off, so a malformed range can't corrupt the slice."""
        keep = 0
        for r in ranges:
            if r.num_symbols <= 0:
                continue
            keep |= ((1 << r.num_symbols) - 1) << r.start_symbol
        return keep & 0x3FFF

    @override
    def _handle_xapp_control(self, dapp_identifier: int, data: bytes):
        dapp_logger.info(f'Triggered control callback for dApp {dapp_identifier}')

        env = self._decode_xapp_control_envelope(data)
        if env["payload_key"] != "prbBlockedControl":
            dapp_logger.info(
                f"xApp control variant {env['type']!r}/{env['payload_key']!r} "
                "not handled by this dApp; dropping"
            )
            return
        prb_blk_list = env["payload"]["blockedPRBs"]
        dapp_logger.info(f"xApp control: blockedPRBs={prb_blk_list}")

        # Route the xApp's absolute list through the same reconciled set as the
        # detection loop. Under the gNB's REPLACE install the two controllers
        # would otherwise clobber each other; reconciling sends the full union.
        self._reconcile_prb_blocks(xapp={int(p) for p in prb_blk_list})
        dapp_logger.info(f"Sending Control to RAN: blacklistedPRBs={prb_blk_list}")

        if self.save_iqs:
            with self._ground_truth_lock:
                ground_truth_label = self._ground_truth_label
            with self._sample_idx_lock:
                if self.sample_idx is not None:
                    self.iq_saver.add_annotation(
                        start_sample=self.sample_idx,
                        label="prb_control",
                        comment=f"Blacklisted {len(prb_blk_list)} PRBs upon message from xApp",
                        timestamp=time.time(),
                        prb_blacklist=prb_blk_list,
                        noise_threshold=self._detector.threshold_db,
                        control_action="blacklist",
                        detector=type(self._detector).__name__,
                        ground_truth_label=ground_truth_label,
                    )
                    dapp_logger.info("Annotation added")

    def _diag_log_full_slot(self, p: SlotPointer, iq_3d: np.ndarray) -> None:
        """One-shot diagnostic over the already-read slot: per-symbol
        energy for antenna 0, plus the bin with the largest magnitude
        in the last symbol. ``iq_3d`` is ``[sym][sc][2]`` float32, the
        slice this dApp processes (antenna 0 of the slot)."""
        n_sym, n_sc, _ = iq_3d.shape
        per_sym_energy = []
        for s in range(n_sym):
            i = iq_3d[s, :, 0]
            q = iq_3d[s, :, 1]
            per_sym_energy.append(float((i * i + q * q).sum()))

        sym = n_sym - 1
        i = iq_3d[sym, :, 0]
        q = iq_3d[sym, :, 1]
        mag = np.sqrt(i * i + q * q)
        top_idx = int(mag.argmax())
        dapp_logger.info(
            f"[DIAG] sfn={p.sfn} slot={p.slot} per-symbol energy: "
            + " ".join(f"s{n}={e:.1f}" for n, e in enumerate(per_sym_energy))
        )
        dapp_logger.info(
            f"[DIAG] sym={sym} max_bin={top_idx} max_mag={mag[top_idx]:.0f} "
            f"mean_mag={mag.mean():.1f} non_zero_bins={int((mag > 1.0).sum())}/{n_sc}"
        )

    def _read_iq_from_shm(self, p: SlotPointer):
        """Read the slot pointed to by ``p`` and return:
          - ``(iq_symbols, iq_3d)``: list of flat float32 [I,Q,I,Q,...]
            arrays (one per symbol of antenna 0) plus a [sym][sc][2]
            view for diagnostics.
        Returns ``None`` if the read fails (out-of-range index, shm not
        open). Antenna 0 is selected to match the legacy convention.
        """
        if not self._ensure_shm_open():
            return None
        full = self._shm_reader.read_slot(p)  # [ant][sym][sc_per_slot][2] float32
        if full is None:
            return None

        # Slice to this dApp's configured PRB span (first num_prbs PRBs
        # of the 273-PRB layout) — trailing PRBs are zero-fill from OAI.
        ant0_full = full[0]                       # [sym][sc_per_slot=3276][2]
        iq_3d = ant0_full[:, : self.ofdm_symbol_size, :]  # [sym][ofdm_symbol_size][2]
        iq_symbols = [iq_3d[s].reshape(-1).copy() for s in range(iq_3d.shape[0])]
        return iq_symbols, iq_3d

    @override
    def _handle_indication(self, dapp_identifier: int, ran_function_id: int, data: bytes):
        """Inbound-thread callback. Dispatch on the RAN function id: the
        L1-KPM SM (RF=2) carries an IQ shm pointer, the Spectrum SM (RF=1)
        carries a Spectrum-SensingIndication. APER has no field tags, so the
        two envelopes are not reliably distinguishable by content (a Spectrum
        indication decodes as a valid L1-KPM one ~25% of the time); the
        ranFunctionIdentifier is the only authoritative discriminant.
        """
        t = Timings(recv_ns=time.monotonic_ns())
        if not data:
            return

        if ran_function_id == self.RAN_FUNCTION_ID:
            p = SlotPointer.from_bytes(data)
            if p is not None:
                self._handle_l1_indication(p, t)
                return
            dapp_logger.info("[SPECTRUM] RF=%d IQ pointer did not decode; dropping",
                             ran_function_id)
            return
        if ran_function_id == self.PRB_CONTROL_RAN_FUNCTION_ID:
            if not self._handle_sensing_indication(data):
                dapp_logger.info("[SPECTRUM] RF=%d sensing indication did not decode; dropping",
                                 ran_function_id)
            return

        dapp_logger.info("[SPECTRUM] indication on unexpected RF=%d; dropping", ran_function_id)

    def _handle_sensing_indication(self, data: bytes) -> bool:
        """Decode an RF=1 Spectrum-SensingIndication and cache its sensing
        ranges (read from the /e3_l2_sensing ring) keyed by (sfn, slot) for the
        RF=2 IQ handler. Returns False if the payload isn't a sensing indication.
        """
        try:
            msg = self._decode_spectrum_message("Spectrum-SensingIndication", data)
        except Exception:
            return False
        try:
            sfn, slot = int(msg["sfn"]), int(msg["slot"])
            if self.encoding_method == "asn1":
                write_idx, n = int(msg["shmWriteIdx"]), int(msg["nRanges"])
            else:
                shm = msg["sensing_shm"]
                write_idx, n = int(shm["write_idx"]), int(shm["n_ranges"])
        except (KeyError, TypeError, ValueError):
            return False

        ranges: tuple = ()
        if n:
            try:
                if not self._sensing_reader.is_open:
                    self._sensing_reader.open()
                ranges = tuple(self._sensing_reader.read_ranges(write_idx, sfn, slot, n))
            except OSError:
                pass
        self._sensing_cache[(sfn, slot)] = ranges
        while len(self._sensing_cache) > 256:
            self._sensing_cache.popitem(last=False)
        return True

    def _shm_staleness_ns(self) -> int:
        """FH-ring TTL derived from the shm header (num_fh_rows × 2 buffers ×
        slot time), falling back to a fixed bound until the header is read."""
        rows = self._shm_reader.num_fh_rows
        if rows:
            return rows * 2 * self._SLOT_DURATION_NS
        return self._SHM_STALENESS_NS_DEFAULT

    def _handle_l1_indication(self, p: SlotPointer, t: Timings) -> None:
        """Read the slot's IQ from /e3_ran_buffers, attach the sensing ranges
        cached from the matching RF=1 indication (by sfn/slot), and queue the
        bundle for the worker."""
        t.decoded_ns = time.monotonic_ns()
        t.producer_ts_ns = p.timestamp_ns

        # Staleness guard. The gNB's FH ring is bounded; if we lag too far
        # behind the producer, the row pointed at by (fh_buffer_index,
        # fh_write_index) has been OVERWRITTEN with a future slot's IQ.
        # The in-band mask is still for the original slot, so we'd display
        # mask-of-slot-N applied to IQ-of-slot-N+k → UE energy bleeding
        # into "sensing" cells.
        #
        # Threshold matches the gNB ring TTL: 64 rows × 2 buffers = 128
        # slots × ~0.5 ms/slot at 30 kHz SCS ≈ 64 ms safety budget. We use
        # 50 ms as a conservative cutoff. If breached, treat the IQ as
        # untrustworthy and skip the slot.
        if p.timestamp_ns > 0:
            lag_ns = t.recv_ns - p.timestamp_ns
            stale_ns = self._shm_staleness_ns()
            if lag_ns > stale_ns:
                self._shm_stale_dropped += 1
                if (self._shm_stale_dropped <= 5
                        or (self._shm_stale_dropped % 500) == 0):
                    dapp_logger.warning(
                        "[SHM] stale slot dropped — lag=%.1f ms > %.0f ms TTL; "
                        "the producer ring has likely rotated past this row. "
                        "Total stale drops: %d.",
                        lag_ns / 1e6, stale_ns / 1e6, self._shm_stale_dropped,
                    )
                return

        after_meta_ns = time.monotonic_ns()
        result = self._read_iq_from_shm(p)
        t.after_meta_ns = after_meta_ns
        if result is None:
            self._count_read_drop()
            return
        iq_symbols, iq_3d = result
        t.after_shm_ns = time.monotonic_ns()

        if (self._shm_indications_handled < 5
                or (self._shm_indications_handled % 500) == 0):
            self._diag_log_full_slot(p, iq_3d)
        self._count_handled(p, iq_symbols)

        # Sensing ranges arrive on RF=1 (Spectrum SM); look up the latest for
        # this (sfn, slot). Empty until the matching RF=1 indication is seen.
        ranges = self._sensing_cache.get((p.sfn, p.slot), ())
        self._tally_sensing(bool(ranges))

        # [MASK] diag: dump ranges per slot for the first 20 then every 200.
        if (self._shm_indications_handled <= 20
                or (self._shm_indications_handled % 200) == 0):
            sample = ", ".join(
                f"(sym {r.start_symbol}+{r.num_symbols}, "
                f"rb {r.rb_start}+{r.rb_size})"
                for r in ranges
            ) or "—"
            dapp_logger.info(
                "[MASK] sfn=%d slot=%d ranges=%d: %s",
                p.sfn, p.slot, len(ranges), sample,
            )

        # Timestamp passed downstream — used by IQSaver annotations.
        # producer_ts_ns is monotonic-ish ns; CSV annotations store it as-is.
        # sfn/slot ride along for diagnostics; ranges drives the
        # worker-thread 2D sensing-window slice.
        self._enqueue_for_worker(
            (iq_symbols, p.timestamp_ns, p.sfn, p.slot, ranges, t)
        )

    def _enqueue_for_worker(self, item) -> None:
        """Hand a fully-read slot to the indication-worker thread. If the
        queue is full, drop the oldest entry so the freshest data wins."""
        try:
            self._ind_queue.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            self._ind_queue.get_nowait()
            self._ind_queue.put_nowait(item)
        except (queue.Empty, queue.Full):
            self._ind_queue_dropped += 1
            return
        self._ind_queue_dropped += 1
        if self._ind_queue_dropped == 1:
            dapp_logger.warning(
                "[SPECTRUM] indication worker queue full; dropping oldest"
            )
        elif (self._ind_queue_dropped % 1000) == 0:
            dapp_logger.warning(
                f"[SPECTRUM] indication worker queue overflow "
                f"(total dropped = {self._ind_queue_dropped})"
            )

    def _indication_worker(self):
        """Drain _ind_queue → mag/detector/dashboard pipeline on pre-read
        data. Lag here only delays display; shm-read already happened on
        the inbound thread so we can't lose data."""
        while not self._ind_worker_stop.is_set():
            try:
                iq_symbols, timestamp, sfn, slot, ranges, t = (
                    self._ind_queue.get(timeout=0.1)
                )
            except queue.Empty:
                continue
            t.dequeue_ns = time.monotonic_ns()
            # Record the max gap between successive successful dequeues
            # since the last stats window.  A 5s freeze in the dashboard
            # shows up here as a 5s gap — the smoking gun for upstream
            # backpressure stalls.  Initialised on first dequeue so the
            # warmup gap isn't misread as a hang.
            if self._ind_worker_last_iter_ns != 0:
                gap_ns = t.dequeue_ns - self._ind_worker_last_iter_ns
                if gap_ns > self._ind_worker_max_gap_ns:
                    self._ind_worker_max_gap_ns = gap_ns
            self._ind_worker_last_iter_ns = t.dequeue_ns
            try:
                self._process_indication(
                    iq_symbols, timestamp, sfn, slot, ranges, t
                )
            except Exception:
                dapp_logger.exception(
                    "[SPECTRUM] _process_indication raised; continuing"
                )

    def _diagnostic_loop(self) -> None:
        """Wall-clock diagnostic thread.  Runs the freeze + flow checks
        once per second so the operator gets a heartbeat even when the
        inbound thread is dead.  The count-driven heartbeat in
        _count_handled (line ~860) is still the primary signal under
        normal operation; this loop is the fallback that catches the
        case where indications stop arriving altogether — gNB crash,
        libe3 connection drop, shm reader stuck on a stale page.
        """
        INTERVAL_S = 1.0
        SILENT_WARN_AFTER_S = 5    # log a WARN after 5s with no new slot
        SILENT_WARN_REPEAT_S = 10  # then once per 10s while still silent
        last_warn_silent_at = 0.0
        while not self._diag_thread_stop.is_set():
            self._diag_thread_stop.wait(INTERVAL_S)
            if self._diag_thread_stop.is_set():
                break
            handled_now = self._shm_indications_handled
            now = time.monotonic()
            if handled_now == self._diag_last_handled:
                self._diag_silent_ticks += 1
            else:
                if self._diag_silent_ticks >= SILENT_WARN_AFTER_S:
                    dapp_logger.info(
                        "[DIAG] indication flow resumed after %d s silence "
                        "(handled=%d)",
                        self._diag_silent_ticks, handled_now,
                    )
                self._diag_silent_ticks = 0
                last_warn_silent_at = 0.0
            self._diag_last_handled = handled_now

            if (self._diag_silent_ticks >= SILENT_WARN_AFTER_S
                    and (now - last_warn_silent_at) >= SILENT_WARN_REPEAT_S):
                last_warn_silent_at = now
                dapp_logger.warning(
                    "[DIAG] no shm indications for %d s — gNB down? libe3 "
                    "transport dead? (handled=%d, queue_drops=%d, qsize=%d)",
                    self._diag_silent_ticks, handled_now,
                    self._ind_queue_dropped, self._ind_queue.qsize(),
                )

    def _build_detector_input(self, mag_batch, ranges):
        """Per-PRB magnitude vector that the threshold detector should
        learn from when sensing_only is on. Builds the same 2D keep mask
        as the dashboard filter, then collapses kept cells per-PRB
        column. Non-sensing PRB columns are set to 0 so the detector
        never sees UE PRBs even when a sensing range partially covers
        the same symbol.

        Returns ``None`` if no sensing cells were present for this slot
        — the caller should skip the detector update entirely to keep
        the adaptive noise floor from drifting on UE bleed.  Otherwise
        returns a tuple ``(out, col_keep)`` where ``out`` is the 1-D
        magnitude vector (n_sc,) and ``col_keep`` is the per-subcarrier
        boolean mask of columns inside the sensing window.  The caller
        uses ``col_keep`` to suppress detections in non-sensing
        columns post hoc — without it, zero-valued non-sensing columns
        get a 0 dB noise floor (after the max(·,1.0) clamp inside the
        detector), which then triggers spurious "interference detected"
        on the UE's actual PRBs.  That would feed back into the
        PRB-blacklist control path and the dApp would happily blacklist
        the UE's own PUSCH allocation.
        """
        if not ranges:
            return None
        n_sym, n_sc = mag_batch.shape
        keep = np.zeros((n_sym, n_sc), dtype=bool)
        for r in ranges:
            if r.num_symbols <= 0 or r.rb_size <= 0:
                continue
            sym_lo = max(0, r.start_symbol)
            sym_hi = min(n_sym, r.start_symbol + r.num_symbols)
            sc_lo  = max(0, r.rb_start * self.num_consecutive_subcarriers_for_prb)
            sc_hi  = min(
                n_sc,
                (r.rb_start + r.rb_size) * self.num_consecutive_subcarriers_for_prb,
            )
            if sym_lo >= sym_hi or sc_lo >= sc_hi:
                continue
            keep[sym_lo:sym_hi, sc_lo:sc_hi] = True
        if not keep.any():
            return None
        # Per-PRB-column mean over kept symbols. Columns where no symbol
        # was kept are zeroed (the per-column denominator drops the
        # contribution by definition). The detector reads a 1-D vector
        # of length n_sc.
        col_keep = keep.any(axis=0)
        col_denom = np.maximum(keep.sum(axis=0).astype(np.float32), 1.0)
        col_sum = np.where(keep, mag_batch, 0.0).sum(axis=0).astype(np.float32)
        out = np.where(col_keep, col_sum / col_denom, 0.0).astype(np.float32)
        return out, col_keep

    def _process_indication(self, iq_symbols, timestamp,
                            sfn: int, slot: int, ranges, t: Timings):
        """Worker-thread processing of an already-shm-read slot. iq_symbols
        is a list of flat float32 [I,Q,I,Q,...] arrays (one per symbol of
        antenna 0), already rescaled by 1/fp16_beta upstream. ``ranges`` is the
        tuple of SensingRange the RF=2 handler looked up in ``_sensing_cache``
        for this (sfn,slot) — cross-SM correlation with the RF=1 Spectrum
        indication that cached them; drives the 2D dashboard filter and the
        detector window when ``self.sensing_only`` is enabled."""
        _ = (sfn, slot)  # reserved for future per-slot diagnostics
        now = time.monotonic()
        n_sym = len(iq_symbols)
        last_iq_arr = iq_symbols[-1]
        dapp_logger.debug(f"iq_symbols: {n_sym} × {last_iq_arr.size} float32")

        if n_sym == 1:
            iq_arr = iq_symbols[0]
            self._I_buf[:] = iq_arr[::2]
            self._Q_buf[:] = iq_arr[1::2]
            np.hypot(self._I_buf, self._Q_buf, out=self._mag_buf)
            mag_batch = self._mag_buf.reshape(1, -1).copy()
        else:
            iq_stack = np.stack(iq_symbols).reshape(n_sym, -1, 2)
            I = iq_stack[..., 0]
            Q = iq_stack[..., 1]
            mag_batch = np.hypot(I, Q)
            self._mag_buf[:] = mag_batch[-1]
        t.after_mag_ns = time.monotonic_ns()

        if self.iqPlotterGui:
            self.iq_queue.put(last_iq_arr)
        if self.dashboard:
            self.demo.publish_slot(mag_batch, sfn, slot,
                                   blocked=self._last_det_mask,
                                   det_thr=self._last_det_thr)
        if self.save_iqs:
            with self._sample_idx_lock:
                self.sample_idx = self.iq_saver.save_samples(last_iq_arr, timestamp=timestamp)
        t.after_dash_ns = time.monotonic_ns()

        # Optional sensing-policy callback.  Fires once per indication
        # (not gated by the detector-interval below) so callbacks driving
        # short cadences (e.g. 10s toggles) see every tick.  Callback
        # decides on its own when to emit; cost of polling is one float
        # compare in the common "not yet" path.
        if self._sensing_policy_callback is not None:
            try:
                cb_result = self._sensing_policy_callback(now)
            except Exception:
                dapp_logger.exception("Error in sensing-policy callback")
                cb_result = None
            if cb_result is not None:
                send_now, mask_per_slot, deactivate = cb_result
                if send_now:
                    self.send_sensing_policy(
                        mask_per_slot=mask_per_slot,
                        deactivate=bool(deactivate),
                    )

        self._detector_run_counter += 1
        if self._detector_run_counter < self._detector_run_interval:
            self._record_latency(t)
            return
        self._detector_run_counter = 0

        # /e3_ran_buffers is already in natural PRB×SC order (no FFT-shift
        # wraparound), so first_carrier_offset is 0 and abs_shifted is just
        # a copy of the magnitudes. Kept as a separate buffer to preserve
        # the downstream detector interface contract.
        #
        # SENSING-WINDOW DETECTOR INPUT: in relaxed sensing_only mode the
        # last symbol of the slot (which is what the legacy code feeds the
        # detector via self._mag_buf = mag_batch[-1]) often contains UE
        # PUSCH or its CP-overlap bleed. The adaptive-threshold detector
        # would then learn UE energy as the "noise floor", flagging real
        # ambient as interference and silently de-tuning. Replace the
        # detector input with a mean over symbols that are actually inside
        # the sensing window when one is available; outside that path the
        # detector keeps its old contract.
        abs_shifted = self._abs_shifted_buf
        # Per-subcarrier mask of columns inside the sensing window. None
        # means "detector ran on the full slot" (legacy path); when set,
        # detections in non-kept columns are suppressed post hoc to
        # avoid feeding back UE PRBs into the PRB-blacklist control.
        det_col_keep = None
        if self.sensing_only:
            if not ranges:
                # No cached sensing window for this (sfn,slot) — the RF=1
                # Spectrum-SensingIndication was lost, late, or absent. Feeding
                # the full slot here would train the noise floor on the UE's own
                # PUSCH, the exact thing sensing_only exists to prevent, so skip
                # the detector entirely and preserve prior state.
                self._record_latency(t)
                return
            det_input = self._build_detector_input(mag_batch, ranges)
            if det_input is None:
                # No sensing-window cells in this slot — don't let the
                # detector learn this update at all (preserves prior state).
                self._record_latency(t)
                return
            keep_for_det, det_col_keep = det_input
            self._abs_shifted_buf[:] = keep_for_det
        else:
            self._abs_shifted_buf[:] = self._mag_buf
        iq_arr = last_iq_arr

        # Detection (strategy-agnostic). det_col_keep (when set) excludes
        # zeroed out-of-window columns from the adaptive noise-floor history.
        ready, blocked, power_db, noise_floor_db = self._detector.update(
            abs_shifted, now, det_col_keep)

        if not ready:
            return

        # Visualization
        if self.energyGui:
            self.sig_queue.put((power_db, noise_floor_db))

        # Suppress detections outside the sensing window.  The detector's
        # noise-floor estimator sees zeros for non-kept columns, which
        # after the 20·log10(max(·,1.0)) clamp gives a 0 dB floor for
        # those bins.  Any real UE energy in those bins then computes a
        # huge SNR and gets flagged as interference — even though those
        # PRBs are PRECISELY the UE's allocation that the operator
        # explicitly carved out of the sensing window.  Without this
        # mask the control path would happily blacklist the UE's own
        # PUSCH PRBs on the basis of poisoned detections.
        if det_col_keep is not None:
            blocked = blocked & det_col_keep

        # Cache for the visualizer: the per-subcarrier detection mask and the
        # active threshold, overlaid on subsequent waterfall frames as the
        # red block strip.
        self._last_det_mask = blocked
        self._last_det_thr = self._detector.threshold_db

        # All PRBs where the detector flagged at least one subcarrier — reported and annotated as-is.
        sc = self.num_consecutive_subcarriers_for_prb
        detected_prbs = np.unique(
            np.where(blocked)[0] // sc
        ).astype(np.uint16)

        dapp_logger.info(
            f"Detected PRBs ({detected_prbs.size}): {detected_prbs.tolist()} | "
            f"detector={type(self._detector).__name__} | "
            f"power_db_max={power_db.max():.1f} dB"
        )

        # Optional sampling-threshold control callback
        update_sampling = False
        if self._sampling_threshold_control_callback is not None:
            try:
                update_sampling, new_sampling_threshold = (
                    self._sampling_threshold_control_callback(detected_prbs, power_db)
                )
                if update_sampling:
                    self.sampling_threshold = new_sampling_threshold
                    dapp_logger.info(
                        f"Custom logic updated sampling threshold to {self.sampling_threshold}"
                    )
                    if self.save_iqs:
                        new_sample_rate = 1 / (0.01 * self.sampling_threshold)
                        self.iq_saver.update_sample_rate(
                            new_sample_rate,
                            sampling_threshold=self.sampling_threshold,
                        )
                        dapp_logger.info(
                            f"Updated IQ saver sample rate to {new_sample_rate:.2f} Hz"
                            f" (sampling_threshold={self.sampling_threshold})"
                        )
            except Exception:
                dapp_logger.exception("Error in custom control callback")
                update_sampling = False

        # Strip DC leakage artefacts from everything: report, annotation, and control.
        dc_low, dc_high = self.DC_LEAKAGE_PRB_LOW, self.DC_LEAKAGE_PRB_HIGH
        detected_prbs = detected_prbs[(detected_prbs < dc_low) | (detected_prbs > dc_high)]
        reported_prbs = detected_prbs
        report_payload = self.create_prb_blacklist_report(
            blacklisted_prbs=reported_prbs.astype(int).tolist()
        )
        # The report carries a Spectrum-DAppReportData envelope, so it must
        # target the Spectrum SM (RF=1) — the L1-KPM SM (RF=2) can't decode it.
        self.e3_interface.schedule_report(
            dappId=self.dapp_id,
            ranFunctionId=self.PRB_CONTROL_RAN_FUNCTION_ID,
            reportData=report_payload,
        )

        if self.control:
            # Filter out BWP/PRACH + guard band — only for the control message to the gNB.
            prb_blk_list = self._compute_prb_blacklist(detected_prbs)
            detected_set = {int(p) for p in prb_blk_list.tolist()}
            # Publish this detection cycle's contribution to the reconciled
            # block set. The gNB install is REPLACE (not additive), so
            # _reconcile_prb_blocks re-sends the FULL union of all sources and
            # never sends a bare delta — a cleared detection set no longer
            # wipes PRBs an operator/xApp still wants blocked, and vice versa.
            self._reconcile_prb_blocks(detect=detected_set, update_sampling=update_sampling)
        else:
            prb_blk_list = np.empty(0, dtype=np.uint16)

        # Annotate the IQ recording with full detection metadata (always, not only when control).
        # Build the annotation dict before acquiring the lock — _build_annotation_fields
        # only reads local variables and immutable detector properties, so it is safe
        # to run outside the critical section, keeping the lock hold-time minimal.
        if self.save_iqs:
            ann = self._build_annotation_fields(
                all_prbs=detected_prbs,
                prb_blk_list=prb_blk_list,
                power_db=power_db,
                noise_floor_db=noise_floor_db,
                timestamp=timestamp,
                comment=(
                    f"Blacklisted {prb_blk_list.size} PRBs due to interference"
                    if self.control
                    else f"Detected {detected_prbs.size} PRBs above threshold (no control active)"
                ),
                control_fired=self.control,
            )
            dapp_logger.info(
                f"Annotation: label={ann['label']} | "
                f"detector={ann.get('detector')} | "
                f"all_detected_prbs={ann['all_detected_prbs']} | "
                f"noise_threshold={ann['noise_threshold']} dB | "
                f"power_db_max={ann['power_db_max']:.1f} dB | "
                f"noise_floor_per_prb={ann.get('noise_floor_per_prb')} | "
                f"snr_db_per_prb={ann.get('snr_db_per_prb')}"
            )
            with self._ground_truth_lock:
                gt = self._ground_truth_label if detected_prbs.size > 0 else "no_rfi"
            ann["ground_truth_label"] = gt
            # Correct for the averaging delay: self.sample_idx is the IQ-sample
            # index of the most recent indication (the last frame of the averaging
            # window). A static detector's decision covers the preceding `window`
            # indications, so the annotation must point back to the first frame of
            # that window. The adaptive detector decides per indication (no window).
            window = (
                self._detector.window
                if isinstance(self._detector, StaticThresholdDetector) else 1
            )
            samples_per_indication = last_iq_arr.size // 2
            with self._sample_idx_lock:
                if self.sample_idx is not None:
                    corrected_start = self._corrected_annotation_start(
                        self.sample_idx, window, samples_per_indication
                    )
                    self.iq_saver.add_annotation(start_sample=corrected_start, **ann)

        t.after_det_ns = time.monotonic_ns()
        self._record_latency(t)

    def _record_latency(self, t: Timings):
        """Append per-stage µs durations; emit a [LATENCY] log every
        _lat_log_interval_s with mean/p50/p99/max per stage."""
        done_ns = time.monotonic_ns()
        lat = self._lat
        if t.producer_ts_ns > 0:
            lat["e2e"].append((t.recv_ns - t.producer_ts_ns) / 1000.0)
        lat["decode"].append((t.decoded_ns - t.recv_ns) / 1000.0)
        lat["meta"].append((t.after_meta_ns - t.decoded_ns) / 1000.0)
        lat["shm"].append((t.after_shm_ns - t.after_meta_ns) / 1000.0)
        lat["qwait"].append((t.dequeue_ns - t.after_shm_ns) / 1000.0)
        lat["mag"].append((t.after_mag_ns - t.dequeue_ns) / 1000.0)
        lat["dash"].append((t.after_dash_ns - t.after_mag_ns) / 1000.0)
        if t.after_det_ns:
            lat["det"].append((t.after_det_ns - t.after_dash_ns) / 1000.0)
        lat["work"].append((done_ns - t.dequeue_ns) / 1000.0)

        now = time.monotonic()
        if now - self._lat_last_log_t < self._lat_log_interval_s:
            return
        if not lat["work"]:
            return
        self._lat_last_log_t = now

        def _stat(deq):
            if not deq:
                return "—"
            arr = np.asarray(deq, dtype=np.float64)
            p50, p99 = np.percentile(arr, [50, 99])
            return f"{arr.mean():.0f}({p50:.0f}/{p99:.0f}/{arr.max():.0f})"

        s = {k: _stat(v) for k, v in lat.items()}
        dapp_logger.info("[LATENCY] n=%d  e2e_µs[mean(p50/p99/max)] = %s",
                         len(lat["work"]), s["e2e"])
        dapp_logger.info(
            "[LATENCY] inbound_µs: decode=%s  meta=%s  shm=%s  |  qwait_µs=%s",
            s["decode"], s["meta"], s["shm"], s["qwait"],
        )
        dapp_logger.info(
            "[LATENCY] worker_µs: mag=%s  dash=%s  det=%s (n_det=%d)  |  work_total=%s",
            s["mag"], s["dash"], s["det"], len(lat["det"]), s["work"],
        )

    @override
    def _control_loop(self):
        """Main-thread loop. Only services GUIs that REQUIRE the main
        thread (matplotlib-based plotters with Tk backend). The dashboard
        is web-based and is driven directly from the indication worker —
        no main-thread hop needed."""
        if not (self.energyGui or self.iqPlotterGui):
            time.sleep(1)
            return

        if self.energyGui:
            try:
                display_data = self.sig_queue.get(timeout=0.1)
                self.energyPlotter.process_iq_data(display_data)
            except queue.Empty:
                pass
            except Exception:
                dapp_logger.exception("[SPECTRUM] Error in energyGui control loop")

        if self.iqPlotterGui:
            try:
                iq_data = self.iq_queue.get(timeout=0.1)
                self.iqPlotter.process_iq_data(iq_data)
            except queue.Empty:
                pass
            except Exception:
                dapp_logger.exception("[SPECTRUM] Error in iqPlotterGui control loop")

    @override
    def _stop(self):
        self._ind_worker_stop.set()
        self._diag_thread_stop.set()
        if self._ind_worker_thread and self._ind_worker_thread.is_alive():
            self._ind_worker_thread.join(timeout=2)
        if self._diag_thread and self._diag_thread.is_alive():
            self._diag_thread.join(timeout=2)
        if self._ind_queue_dropped:
            dapp_logger.info(
                f"[SPECTRUM] indication worker queue dropped "
                f"{self._ind_queue_dropped} entries total"
            )

        if self.save_iqs:
            self.iq_saver.close()
        if self.dashboard:
            self.demo.stop()

        total_sent = self._shm_indications_handled + self._shm_indications_dropped
        denom = max(1, total_sent)
        dapp_logger.info(
            "[SHM] final stats: handled=%d read_drops=%d (%.2f%%) total=%d",
            self._shm_indications_handled,
            self._shm_indications_dropped,
            100.0 * self._shm_indications_dropped / denom,
            total_sent,
        )

        denom_sensing = max(1, self._sensing_slots_total)
        dapp_logger.info(
            "[SENSING] final stats: L1_slots=%d with_ranges=%d (%.2f%%) "
            "sensing_only=%s strict_sensing=%s strict_dropped=%d",
            self._sensing_slots_total, self._sensing_slots_with_ranges,
            100.0 * self._sensing_slots_with_ranges / denom_sensing,
            self.sensing_only, self.strict_sensing, self._strict_dropped,
        )

        try:
            self._shm_reader.close()
            self._sensing_reader.close()
        except Exception as exc:
            dapp_logger.debug(f"[SHM] reader close failed: {exc}")
