#!/usr/bin/env python3
"""A PRB-granularity detector runs through _process_indication with the same
(per-subcarrier) outputs as the subcarrier one, and a disabled detector skips."""
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectrum.spectrum_dapp import SpectrumSharingDApp, Timings
from spectrum.threshold_detector import StaticThresholdDetector

NUM_PRBS = 8
FFT = NUM_PRBS * 12


class _Iface:
    def __init__(self):
        self.reports = []

    def schedule_report(self, **kw):
        self.reports.append(kw)


def _dapp(detector):
    d = object.__new__(SpectrumSharingDApp)
    d.dapp_id = 1
    d.num_prbs = NUM_PRBS
    d.fft_size = FFT
    d.ofdm_symbol_size = FFT
    d.num_consecutive_subcarriers_for_prb = 12
    d.DC_LEAKAGE_PRB_LOW, d.DC_LEAKAGE_PRB_HIGH = 100, 100
    d._detector = detector
    d._detector_lock = threading.Lock()
    d._det_enabled = True
    d._det_prb_mode = False
    d._last_det_mask = None
    d._last_det_thr = None
    d.control = False
    d.sensing_only = False
    d.save_iqs = False
    d.dashboard = False
    d.energyGui = False
    d.iqPlotterGui = False
    d._sensing_policy_callback = None
    d._sampling_threshold_control_callback = None
    d._detector_run_interval = 1
    d._detector_run_counter = 0
    d._I_buf = np.empty(FFT, np.float32)
    d._Q_buf = np.empty(FFT, np.float32)
    d._mag_buf = np.empty(FFT, np.float32)
    d._abs_shifted_buf = np.empty(FFT, np.float32)
    d.e3_interface = _Iface()
    d.create_prb_blacklist_report = lambda blacklisted_prbs: bytes(blacklisted_prbs)
    d._record_latency = lambda t: None
    return d


def _slot():
    iq = np.zeros(FFT * 2, dtype=np.float32)
    iq[0::2] = 10.0          # I = 10 everywhere (20 dB)
    iq[2 * 12 * 3:2 * 12 * 4:2] = 5000.0   # PRB 3 hot
    return [iq]


def _run(d):
    d._process_indication(_slot(), 1.0, 1, 1, (), Timings())


def test_prb_mode_matches_subcarrier_mode():
    outs = {}
    for prb in (False, True):
        d = _dapp(StaticThresholdDetector(40.0, FFT, window=1))
        if prb:
            d._detector = d._detector.resized(NUM_PRBS)
            d._det_prb_mode = True
        _run(d)
        outs[prb] = (d._last_det_mask, list(d.e3_interface.reports[-1]["reportData"]))
    sc_mask, sc_prbs = outs[False]
    prb_mask, prb_prbs = outs[True]
    assert sc_mask.shape == prb_mask.shape == (FFT,)
    assert sc_prbs == [3]
    assert prb_prbs == [3]
    assert prb_mask.reshape(NUM_PRBS, 12).all(axis=1).tolist() == [False, False, False, True] + [False] * 4


def test_disabled_detector_skips():
    d = _dapp(StaticThresholdDetector(40.0, FFT, window=1))
    d._det_enabled = False
    _run(d)
    assert d.e3_interface.reports == []


# --- L1xL2 filter masks what the dashboard shows ---------------------------

from spectrum.e3_l2_sensing_reader import SensingRange


class _Demo:
    def __init__(self):
        self.frames = []

    def publish_slot(self, mag, sfn, slot, blocked=None, det_thr=None):
        self.frames.append(np.array(mag))


def _slot14():
    """14 symbols, magnitude 10 everywhere, PRB 5 hot (5000)."""
    out = []
    for _ in range(14):
        iq = np.zeros(FFT * 2, dtype=np.float32)
        iq[0::2] = 10.0
        iq[0:FFT * 2:2][60:72] = 5000.0
        out.append(iq)
    return out


def _with_dashboard(sensing_only):
    d = _dapp(StaticThresholdDetector(1000.0, FFT, window=1))
    d.dashboard = True
    d.demo = _Demo()
    d.sensing_only = sensing_only
    d._detector_run_interval = 4
    return d


def test_filter_on_publishes_only_the_sensed_cells():
    d = _with_dashboard(True)
    ranges = (SensingRange(start_symbol=8, num_symbols=1, rb_start=0, rb_size=4),)
    d._process_indication(_slot14(), 1.0, 1, 8, ranges, Timings())
    frame = d.demo.frames[-1]
    assert frame.shape == (14, FFT)
    keep = np.zeros((14, FFT), dtype=bool)
    keep[8, :48] = True
    assert (frame[keep] == 10.0).all()
    assert (frame[~keep] == 0.0).all()          # the hot PRB 5 is masked, not shown
    assert frame[:, 60:72].max() == 0.0


def test_filter_off_publishes_the_raw_slot():
    d = _with_dashboard(False)
    ranges = (SensingRange(start_symbol=8, num_symbols=1, rb_start=0, rb_size=4),)
    d._process_indication(_slot14(), 1.0, 1, 8, ranges, Timings())
    assert d.demo.frames[-1][:, 60:72].min() == 5000.0


def test_valid_symbol_mask_masks_the_dashboard_too():
    d = _with_dashboard(True)
    ranges = (SensingRange(start_symbol=0, num_symbols=14, rb_start=0, rb_size=NUM_PRBS),)
    d._process_indication(_slot14(), 1.0, 1, 8, ranges, Timings(), valid_symbol_mask=1 << 8)
    frame = d.demo.frames[-1]
    assert (frame[8] > 0).all() and (np.delete(frame, 8, axis=0) == 0).all()


def test_slot_without_sensing_ranges_is_not_published():
    d = _with_dashboard(True)
    d._process_indication(_slot14(), 1.0, 1, 8, (), Timings())
    assert d.demo.frames == []


def test_each_frame_uses_its_own_ranges_between_detector_runs():
    d = _with_dashboard(True)           # detector runs every 4th slot
    for i, rb in enumerate((0, 2, 4)):
        ranges = (SensingRange(start_symbol=8, num_symbols=1, rb_start=rb, rb_size=2),)
        d._process_indication(_slot14(), 1.0, 1, 8 + i, ranges, Timings())
    assert d.e3_interface.reports == []  # the detector has not run yet
    for frame, rb in zip(d.demo.frames, (0, 2, 4)):
        cols = np.where(frame[8] > 0)[0]
        assert cols.min() == rb * 12 and cols.max() == (rb + 2) * 12 - 1
