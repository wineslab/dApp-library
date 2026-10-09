#!/usr/bin/env python3
"""In-place parameter changes of the threshold detectors."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectrum.threshold_detector import AdaptiveThresholdDetector, StaticThresholdDetector


def test_adaptive_configure_in_place():
    d = AdaptiveThresholdDetector(20.0, 24, hist_depth=4, embargo_timeout_secs=1.0)
    d.configure(snr_threshold_db=15.0, embargo_timeout_secs=2.5)
    assert d.threshold_db == 15.0 and d.embargo_timeout_secs == 2.5 and d.hist_depth == 4


def test_adaptive_hist_depth_discards_history():
    d = AdaptiveThresholdDetector(20.0, 24, hist_depth=2)
    mag = np.full(24, 100.0, dtype=np.float32)
    for i in range(3):
        d.update(mag, float(i))
    assert d._noise_floor_est.is_ready
    d.configure(hist_depth=8)
    assert d.hist_depth == 8 and not d._noise_floor_est.is_ready
    d.configure(hist_depth=8)  # same depth keeps state
    assert d.hist_depth == 8


def test_adaptive_validates_before_applying():
    d = AdaptiveThresholdDetector(20.0, 24, hist_depth=4, embargo_timeout_secs=1.0)
    with pytest.raises(ValueError):
        d.configure(snr_threshold_db=99.0, hist_depth=0)
    assert d.threshold_db == 20.0
    with pytest.raises(ValueError):
        d.configure(snr_threshold_db=99.0, embargo_timeout_secs=-1)
    assert d.threshold_db == 20.0


def test_static_window_change_restarts_accumulation():
    d = StaticThresholdDetector(40.0, 12, window=4)
    mag = np.full(12, 1000.0, dtype=np.float32)
    for i in range(3):
        assert d.update(mag, float(i))[0] is False
    d.configure(window=2)
    assert d._count == 0 and d.window == 2
    assert d.update(mag, 10.0)[0] is False
    ready, blocked, *_ = d.update(mag, 11.0)
    assert ready and blocked.all()
    d.configure(threshold_db=100.0)
    assert d.threshold_db == 100.0


def test_resized_keeps_parameters_with_fresh_state():
    a = AdaptiveThresholdDetector(18.0, 24, hist_depth=5, embargo_timeout_secs=3.0).resized(2)
    assert (a.fft_size, a.threshold_db, a.hist_depth, a.embargo_timeout_secs) == (2, 18.0, 5, 3.0)
    s = StaticThresholdDetector(33.0, 24, window=7).resized(2)
    assert (s.fft_size, s.threshold_db, s.window) == (2, 33.0, 7)
