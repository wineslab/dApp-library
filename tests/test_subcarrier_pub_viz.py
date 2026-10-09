#!/usr/bin/env python3
"""Server-side visualizer publish controls (granularity, aggregation, subsample)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytest.importorskip("zmq")

import threading

from visualization.subcarrier_pub import SubcarrierPublisher

N_PRB, N_SYM = 4, 14


class _Sock:
    def __init__(self):
        self.sent = []

    def send_multipart(self, parts, flags=0):
        self.sent.append(parts)


def _pub():
    p = object.__new__(SubcarrierPublisher)
    p._db_min, p._db_max, p._span = 20, 90, 70.0
    p._num_prbs = N_PRB
    p._viz_cfg = ("sc", "sym", "mean", 1)
    p._viz_lock = threading.Lock()
    p._slot_counter = 0
    p._sock = _Sock()
    return p


def _meta(parts):
    f = parts[0].decode().split("|")
    return {"n_sym": int(f[4]), "n_sc": int(f[5]), "det_n": int(f[10].split("=")[1]),
            "det": int(f[9].split("=")[1])}


def _grid():
    m = np.full((N_SYM, N_PRB * 12), 100.0, dtype=np.float32)
    m[:, 12:24] = 10000.0
    return m


def test_default_publishes_full_grid():
    p = _pub()
    p.publish_slot(_grid(), 1, 2)
    meta = _meta(p._sock.sent[0])
    assert (meta["n_sym"], meta["n_sc"]) == (N_SYM, N_PRB * 12)


@pytest.mark.parametrize("freq,time_,shape", [
    ("prb", "sym", (N_SYM, N_PRB)),
    ("sc", "slot", (1, N_PRB * 12)),
    ("prb", "slot", (1, N_PRB)),
])
def test_granularity_shapes(freq, time_, shape):
    p = _pub()
    p.set_viz_config(freq_gran=freq, time_gran=time_)
    p.publish_slot(_grid(), 1, 2)
    parts = p._sock.sent[0]
    meta = _meta(parts)
    assert (meta["n_sym"], meta["n_sc"]) == shape
    assert len(parts[1]) == shape[0] * shape[1]


def test_mean_vs_max_aggregation():
    p = _pub()
    m = np.full((N_SYM, N_PRB * 12), 100.0, dtype=np.float32)
    m[0, 0] = 100000.0
    p.set_viz_config(freq_gran="prb", time_gran="slot", agg="mean")
    p.publish_slot(m, 1, 2)
    mean_v = p._sock.sent[-1][1][0]
    p.set_viz_config(agg="max")
    p.publish_slot(m, 1, 3)
    max_v = p._sock.sent[-1][1][0]
    assert max_v > mean_v


def test_detection_mask_follows_frequency_granularity():
    p = _pub()
    mask = np.zeros(N_PRB * 12, dtype=bool)
    mask[12 + 3] = True
    p.set_viz_config(freq_gran="prb")
    p.publish_slot(_grid(), 1, 2, blocked=mask, det_thr=30.0)
    parts = p._sock.sent[0]
    assert _meta(parts)["det_n"] == N_PRB
    assert list(parts[2]) == [0, 1, 0, 0]


def test_subsample_keeps_one_in_n():
    p = _pub()
    p.set_viz_config(subsample_n=3)
    for i in range(9):
        p.publish_slot(_grid(), 1, i)
    assert len(p._sock.sent) == 3


def test_invalid_config_changes_nothing():
    p = _pub()
    with pytest.raises(ValueError):
        p.set_viz_config(freq_gran="prb", subsample_n=0)
    with pytest.raises(ValueError):
        p.set_viz_config(agg="median")
    assert p.get_viz_config() == {"freq_gran": "sc", "time_gran": "sym", "agg": "mean", "subsample_n": 1}
