#!/usr/bin/env python3
"""The dashboard page follows --num-prbs and exposes the control capabilities."""
import importlib.util
import os
import re
import sys

import pytest

pytest.importorskip("flask")
pytest.importorskip("flask_sock")
pytest.importorskip("zmq")

_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "visualization",
                     "subcarrier", "subcarrier_visualizer.py")


@pytest.fixture
def viz(monkeypatch):
    spec = importlib.util.spec_from_file_location("subcarrier_visualizer_under_test", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


def test_page_has_no_fixed_273_grid_literals(viz):
    viz.VIZ_NUM_PRBS = 106
    html = viz.app.test_client().get("/").get_data(as_text=True)
    assert re.search(r"const N_SC\s*=\s*1272;", html) and "N_PRBS = 106" in html
    visible = re.sub(r"<!--.*?-->|//[^\n]*|/\*.*?\*/", "", html, flags=re.S)
    for literal in ("273", "3276", "3275", "272"):
        assert not re.search(rf"\b{literal}\b", visible), literal
    assert "106 PRBs" in html and "subcarrier (1272)" in html and "PRB (106)" in html


def test_page_default_grid_is_273(viz):
    html = viz.app.test_client().get("/").get_data(as_text=True)
    assert "273 PRBs" in html and "3275" in html


def test_caps_route_reports_disabled_control(viz):
    viz._ctrl_enabled = False
    r = viz.app.test_client().get("/ctrl/caps").get_json()
    assert r["ok"] is False and "not available" in r["error"]


def test_caps_route_proxies_to_the_dapp(viz, monkeypatch):
    seen = []
    monkeypatch.setattr(viz, "_ctrl_request", lambda p, timeout_ms=2000: seen.append(p) or {"ok": True, "dapp": "python"})
    assert viz.app.test_client().get("/ctrl/caps").get_json()["dapp"] == "python"
    assert seen == [{"cmd": "caps"}]


def test_page_labels_absolute_frequencies(viz):
    viz.VIZ_NUM_PRBS = 106
    viz.VIZ_CENTER_FREQ_HZ = 3619.2e6
    viz.VIZ_SCS_KHZ = 30.0
    html = viz.app.test_client().get("/").get_data(as_text=True)
    assert "centre 3619.20 MHz" in html
    # Left edge = centre - 106*12*30 kHz / 2; PRB 72 is centred at 3626.22 MHz.
    m = re.search(r"F_LEFT_MHZ = ([0-9.]+)", html)
    assert abs(float(m.group(1)) - 3600.12) < 1e-6
    assert abs(float(m.group(1)) + (72 * 12 + 6) * 0.03 - 3626.22) < 1e-6


def test_page_without_carrier_frequency_keeps_indices_only(viz):
    html = viz.app.test_client().get("/").get_data(as_text=True)
    assert "F_LEFT_MHZ = null" in html and "centre" not in html.split("<header")[1].split("</header>")[0]
