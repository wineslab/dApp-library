#!/usr/bin/env python3
"""Dashboard control protocol on a dApp built without a gNB."""
import os
import sys
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectrum.control_handlers import ControlHandlers
from spectrum.policy import BaselinePolicy, PolicyDetector
from spectrum.slot_recorder import SlotRecorder
from spectrum.spectrum_dapp import SpectrumSharingDApp
from spectrum.threshold_detector import AdaptiveThresholdDetector, StaticThresholdDetector

NUM_PRBS = 6
FFT = NUM_PRBS * 12


class _FakeIface:
    def __init__(self):
        self.controls = []

    def schedule_control(self, *, dappId, ranFunctionId, controlId, actionData, sequenceId=0):
        self.controls.append(actionData)


class _FakeSaver:
    def __init__(self, **kw):
        self.kw = kw
        self.samples = 0
        self.annotations = []
        self.closed = False

    def save_samples(self, data, timestamp=None):
        start = self.samples
        self.samples += len(data)
        return start

    def add_annotation(self, **kw):
        self.annotations.append(kw)

    def get_recording_info(self):
        return {"file_paths": ["/tmp/x.sigmf-data"],
                "file_size_bytes": {"/tmp/x.sigmf-data": self.samples * 8}}

    def close(self):
        self.closed = True


def _dapp(detector, control=True, tmp_path=None, savers=None):
    d = object.__new__(SpectrumSharingDApp)
    d.dapp_id = 1
    d.encoding_method = "asn1"
    d.num_prbs = NUM_PRBS
    d.fft_size = FFT
    d.num_consecutive_subcarriers_for_prb = 12
    d._detector = detector
    d._detector_lock = threading.Lock()
    d._det_enabled = True
    d._det_prb_mode = False
    d._last_det_mask = None
    d.sensing_only = True
    d.control = control
    d._detect_blocks_enabled = control
    d._prb_block_lock = threading.Lock()
    d._prb_block_detect = set()
    d._prb_block_xapp = set()
    d._prb_block_sent = set()
    d.e3_interface = _FakeIface()
    d.create_prb_block_control = lambda blocked_prbs, update_sampling=False: repr(blocked_prbs).encode()
    d.dashboard = False
    d._recorder = SlotRecorder(
        base_path=str(tmp_path or "/tmp"), center_freq=3.6e9, fp16_beta=1 / 2048,
        saver_factory=lambda **kw: (savers.append(_FakeSaver(**kw)) or savers[-1]) if savers is not None else _FakeSaver(**kw))
    return d


def _sent(d):
    return [set(eval(a.decode())) for a in d.e3_interface.controls]


def _adaptive():
    return AdaptiveThresholdDetector(30.0, FFT, hist_depth=32, embargo_timeout_secs=9.9)


# --- block ---------------------------------------------------------------

def test_block_off_retracts_detection_but_keeps_xapp():
    d = _dapp(_adaptive())
    h = ControlHandlers(d)
    d._reconcile_prb_blocks(detect={1, 2})
    d._reconcile_prb_blocks(xapp={5})
    assert _sent(d)[-1] == {1, 2, 5}
    r = h.cmd_block({"cmd": "block", "enabled": False})
    assert r == {"ok": True, "enabled": False, "manual": False, "manual_count": 0}
    assert _sent(d)[-1] == {5}
    # A detection cycle already in flight cannot re-install after the off.
    d._reconcile_prb_blocks(detect={1, 2})
    assert d._prb_block_sent == {5}
    h.cmd_block({"cmd": "block", "enabled": True})
    d._reconcile_prb_blocks(detect={1})
    assert _sent(d)[-1] == {1, 5}


def test_block_query_and_manual_lists():
    d = _dapp(_adaptive(), control=False)
    h = ControlHandlers(d)
    assert h.cmd_block({"cmd": "block"})["enabled"] is False
    with pytest.raises(ValueError, match="manual PRB lists"):
        h.cmd_block({"cmd": "block", "prbs": [1]})
    with pytest.raises(ValueError, match="boolean"):
        h.cmd_block({"cmd": "block", "enabled": "yes"})


def test_block_toggle_vs_worker_race():
    d = _dapp(_adaptive())
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            d._reconcile_prb_blocks(detect={3, 4})

    t = threading.Thread(target=worker)
    t.start()
    try:
        for _ in range(200):
            d.set_block_enabled(False)
            d.set_block_enabled(True)
        d.set_block_enabled(False)
    finally:
        stop.set()
        t.join()
    assert d._prb_block_sent == set() and _sent(d)[-1] == set()


# --- detect --------------------------------------------------------------

def test_detect_adaptive_roundtrip_and_query():
    d = _dapp(_adaptive())
    h = ControlHandlers(d)
    r = h.cmd_detect({"cmd": "detect", "config": {
        "threshold_db": 22.5, "hist_depth": 8, "embargo_secs": 2.0, "mask_with_sensing": False}})
    assert r["ok"] and r["threshold_db"] == 22.5 and r["hist_depth"] == 8
    assert r["embargo_secs"] == 2.0 and r["mask_with_sensing"] is False and d.sensing_only is False
    q = h.cmd_detect({"cmd": "detect", "config": {}})
    assert {k: q[k] for k in ("threshold_db", "hist_depth", "embargo_secs")} == {
        "threshold_db": 22.5, "hist_depth": 8, "embargo_secs": 2.0}


def test_detect_is_all_or_nothing():
    d = _dapp(_adaptive())
    h = ControlHandlers(d)
    with pytest.raises(ValueError):
        h.cmd_detect({"cmd": "detect", "config": {"threshold_db": 99, "hist_depth": 0}})
    with pytest.raises(ValueError):
        h.cmd_detect({"cmd": "detect", "config": {"granularity": "prb", "embargo_secs": -1}})
    assert d._detector.threshold_db == 30.0 and d._det_prb_mode is False
    with pytest.raises(ValueError, match="granularity"):
        h.cmd_detect({"cmd": "detect", "config": {"granularity": "bin"}})


def test_detect_granularity_switch_rebuilds_with_matching_size():
    d = _dapp(_adaptive())
    h = ControlHandlers(d)
    r = h.cmd_detect({"cmd": "detect", "config": {"granularity": "prb"}})
    assert r["granularity"] == "prb" and r["n_bins"] == NUM_PRBS
    assert d._detector.fft_size == NUM_PRBS and d._detector.threshold_db == 30.0
    r = h.cmd_detect({"cmd": "detect", "config": {"granularity": "sc"}})
    assert r["n_bins"] == FFT and d._detector.fft_size == FFT


def test_detect_disable_retracts_detection_blocks():
    d = _dapp(_adaptive())
    h = ControlHandlers(d)
    d._reconcile_prb_blocks(detect={2})
    h.cmd_detect({"cmd": "detect", "config": {"enabled": False}})
    assert d._det_enabled is False and d._prb_block_sent == set()
    d._reconcile_prb_blocks(detect={2})  # in-flight cycle is ignored
    assert d._prb_block_sent == set()
    h.cmd_detect({"cmd": "detect", "config": {"enabled": True}})
    d._reconcile_prb_blocks(detect={2})
    assert d._prb_block_sent == {2}


def test_detect_static_has_no_embargo_but_window():
    d = _dapp(StaticThresholdDetector(40.0, FFT, window=64))
    h = ControlHandlers(d)
    with pytest.raises(ValueError, match="embargo"):
        h.cmd_detect({"cmd": "detect", "config": {"embargo_secs": 3}})
    r = h.cmd_detect({"cmd": "detect", "config": {"hist_depth": 16, "threshold_db": 45}})
    assert r["hist_depth"] == 16 and r["threshold_db"] == 45 and d._detector.window == 16


def test_detect_under_policy_only_accepts_the_sensing_filter():
    det = PolicyDetector(BaselinePolicy(), fft_size=FFT, num_prbs=NUM_PRBS)
    d = _dapp(det)
    h = ControlHandlers(d)
    with pytest.raises(ValueError, match="policy"):
        h.cmd_detect({"cmd": "detect", "config": {"enabled": False}})
    # The dashboard sends every field on a change; unchanged ones are no-ops.
    cur = d.detector_state()
    r = h.cmd_detect({"cmd": "detect", "config": {
        "enabled": True, "granularity": "sc", "threshold_db": cur["threshold_db"],
        "hist_depth": cur["hist_depth"], "embargo_secs": cur["embargo_secs"],
        "mask_with_sensing": False}})
    assert r["ok"] and d.sensing_only is False and r["detector"] == "policy"


# --- caps ----------------------------------------------------------------

def test_caps_matrix():
    h = ControlHandlers(_dapp(_adaptive()))
    c = h.cmd_caps({})
    assert c["dapp"] == "python" and c["proto"] == 1 and c["n_prbs"] == NUM_PRBS
    assert all(c["detect"]["fields"].values()) and c["detect"]["detector"] == "adaptive"
    assert c["record"]["supported"] is True and c["viz"]["supported"] is False

    s = ControlHandlers(_dapp(StaticThresholdDetector(40.0, FFT))).cmd_caps({})["detect"]
    assert s["fields"]["embargo_secs"] is False and s["fields"]["hist_depth"] is True

    p = ControlHandlers(_dapp(PolicyDetector(BaselinePolicy(), FFT, NUM_PRBS))).cmd_caps({})["detect"]
    assert p["detector"] == "policy" and p["reason"]
    assert [k for k, v in p["fields"].items() if v] == ["mask_with_sensing"]


# --- viz -----------------------------------------------------------------

def test_viz_requires_a_dashboard():
    h = ControlHandlers(_dapp(_adaptive()))
    with pytest.raises(ValueError, match="no dashboard"):
        h.cmd_viz({"cmd": "viz", "config": {}})


# --- recording -----------------------------------------------------------

def _slot(n_ant=2):
    return np.ones((n_ant, 14, 3276, 2), dtype=np.float32)


def _wait(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_recording_lifecycle(tmp_path):
    savers = []
    d = _dapp(_adaptive(), tmp_path=tmp_path, savers=savers)
    h = ControlHandlers(d)
    assert h.cmd_status({})["state"] == "idle"
    with pytest.raises(ValueError, match="not recording"):
        h.cmd_stop({})
    r = h.cmd_start({"cmd": "start", "duration_s": 30, "notes": "tone on PRB 3"})
    assert r["ok"] and r["filename"].startswith("iq_full_")
    assert savers[0].kw["dtype"] == "cf32_le" and savers[0].kw["user_notes"] == "tone on PRB 3"
    with pytest.raises(ValueError, match="already recording"):
        h.cmd_start({"cmd": "start", "duration_s": 5})

    class R:
        start_symbol, num_symbols, rb_start, rb_size = 8, 1, 0, 6
    for i in range(3):
        d._recorder.offer(_slot(), 1 + i, 10, i, (R,), 0x3FFF)
    assert _wait(lambda: h.cmd_status({})["slots"] == 3)
    s = h.cmd_stop({})
    assert s["ok"] and s["slots"] == 3 and s["finalized"] is True
    assert s["bytes"] == 3 * 2 * 14 * 3276 * 8
    assert savers[0].closed and len(savers[0].annotations) == 3
    assert savers[0].annotations[0]["sensing_ranges"][0]["rb_size"] == 6
    assert savers[0].annotations[0]["n_ants"] == 2
    assert savers[0].annotations[0]["timestamp_ns"] == 1 and "timestamp" not in savers[0].annotations[0]
    # Offers while idle are ignored.
    d._recorder.offer(_slot(), 9.0, 1, 1, (), 0x3FFF)
    assert h.cmd_status({})["state"] == "idle"


def test_recording_stops_when_the_disk_fills(tmp_path, monkeypatch):
    import spectrum.slot_recorder as sr
    monkeypatch.setattr(sr, "_DISK_CHECK_EVERY", 2)
    savers = []
    d = _dapp(_adaptive(), tmp_path=tmp_path, savers=savers)
    d._recorder.start(60, "")
    monkeypatch.setattr(sr.shutil, "disk_usage", lambda p: type("U", (), {"free": 1})())
    for i in range(4):
        d._recorder.offer(_slot(1), 1.0 + i, 1, i, (), 0x3FFF)
    assert _wait(lambda: savers[0].closed)
    st = d._recorder.status()
    assert st["state"] == "idle" and st["filename"] == "x.sigmf-data"


def test_recording_duration_expiry_and_validation(tmp_path):
    savers = []
    d = _dapp(_adaptive(), tmp_path=tmp_path, savers=savers)
    h = ControlHandlers(d)
    for bad in (0, -1, 86401, "x"):
        with pytest.raises(ValueError):
            h.cmd_start({"cmd": "start", "duration_s": bad})
    with pytest.raises(ValueError):
        h.cmd_start({"cmd": "start", "duration_s": 5, "notes": "x" * (16 * 1024 + 1)})
    h.cmd_start({"cmd": "start", "duration_s": 0.05})
    time.sleep(0.1)
    d._recorder.tick()
    assert _wait(lambda: savers[0].closed)
    assert h.cmd_status({})["state"] == "idle"


def test_recording_unavailable_is_reported(tmp_path):
    d = _dapp(_adaptive(), tmp_path=tmp_path)
    d._recorder = SlotRecorder(base_path=str(tmp_path), center_freq=0, fp16_beta=1.0)
    import builtins
    real = builtins.__import__

    def deny(name, *a, **k):
        if name == "iq_saver.iq_saver":
            raise ImportError("no native")
        return real(name, *a, **k)

    builtins.__import__ = deny
    try:
        caps = ControlHandlers(d).cmd_caps({})
        assert caps["record"]["supported"] is False and "native" in caps["record"]["reason"]
        with pytest.raises(ValueError):
            ControlHandlers(d).cmd_start({"cmd": "start", "duration_s": 5})
    finally:
        builtins.__import__ = real
