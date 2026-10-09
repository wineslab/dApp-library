"""Dashboard control protocol (v1) for the spectrum dApp.

Maps the dashboard's ``block`` / ``detect`` / ``viz`` / recording requests and
the ``caps`` query onto :class:`spectrum.spectrum_dapp.SpectrumSharingDApp`.
The transport is :class:`dapp.control_server.ControlServer`; a handler returns
the reply dict or raises ``ValueError(reason)``, which becomes
``{"ok": false, "error": reason}``.
"""

from __future__ import annotations

import math

PROTO_VERSION = 1
MAX_DURATION_S = 86400.0
MAX_NOTES_BYTES = 16 * 1024

_DETECT_KEYS = ("enabled", "granularity", "threshold_db", "hist_depth",
                "embargo_secs", "mask_with_sensing")


def _config(req: dict) -> dict:
    """Requests carry their fields nested under ``config`` or at top level."""
    cfg = req.get("config", req)
    if not isinstance(cfg, dict):
        raise ValueError("'config' must be an object")
    return cfg


def _bool(cfg: dict, key: str):
    v = cfg.get(key)
    if v is not None and not isinstance(v, bool):
        raise ValueError(f"{key} must be a boolean")
    return v


class ControlHandlers:
    def __init__(self, dapp) -> None:
        self._d = dapp

    # ------------------------------------------------------------------

    def cmd_caps(self, req: dict) -> dict:
        d = self._d
        kind = d.detector_kind()
        every = {k: True for k in _DETECT_KEYS}
        if kind == "policy":
            fields = {k: k == "mask_with_sensing" for k in _DETECT_KEYS}
            reason = "a policy decides the blocks; detector parameters are set in the policy"
        elif kind == "static":
            fields = dict(every, embargo_secs=False)
            reason = "static detector has no embargo"
        elif kind == "adaptive":
            fields, reason = every, ""
        else:
            fields = {k: k == "mask_with_sensing" for k in _DETECT_KEYS}
            reason = "detector parameters do not apply"
        rec_ok, rec_why = d._recorder.available()
        has_viz = bool(getattr(d, "dashboard", False))
        return {
            "ok": True,
            "dapp": "python",
            "proto": PROTO_VERSION,
            "n_prbs": d.num_prbs,
            "block": {"supported": True, "enabled": bool(d.control), "reason": ""},
            "detect": {"supported": True, "detector": kind,
                       "fields": fields, "reason": reason},
            "viz": {"supported": has_viz,
                    "reason": "" if has_viz else "no dashboard publisher in this run"},
            "record": {"supported": rec_ok, "reason": rec_why},
        }

    def cmd_block(self, req: dict) -> dict:
        d = self._d
        if "prbs" in req or "range" in req:
            raise ValueError("manual PRB lists not supported")
        enabled = _bool(req, "enabled")
        if enabled is not None:
            d.set_block_enabled(enabled)
        return {"ok": True, "enabled": bool(d.control), "manual": False,
                "manual_count": 0}

    def cmd_detect(self, req: dict) -> dict:
        d = self._d
        cfg = {k: cfg_v for k, cfg_v in _config(req).items() if k in _DETECT_KEYS}
        enabled = _bool(cfg, "enabled")
        mask = _bool(cfg, "mask_with_sensing")
        gran = cfg.get("granularity")
        # The dashboard sends all six fields on every change; a field that
        # already has the requested value is not a change.
        cur = d.detector_state()
        fields = {
            "enabled": enabled,
            "granularity": gran,
            "threshold_db": cfg.get("threshold_db"),
            "hist_depth": cfg.get("hist_depth"),
            "embargo_secs": cfg.get("embargo_secs"),
        }
        for k, v in list(fields.items()):
            if v is None:
                continue
            c = cur[k]
            try:
                same = (math.isclose(float(v), float(c), abs_tol=1e-9)
                        if isinstance(c, float) else v == c)
            except (TypeError, ValueError):
                same = False
            if same:
                fields[k] = None
        state = d.configure_detector(mask_with_sensing=mask, **fields)
        return {"ok": True, **state}

    def cmd_viz(self, req: dict) -> dict:
        d = self._d
        pub = getattr(d, "demo", None) if getattr(d, "dashboard", False) else None
        if pub is None:
            raise ValueError("no dashboard publisher in this run")
        cfg = _config(req)
        sub = cfg.get("subsample_n")
        if sub is not None and (isinstance(sub, bool) or not isinstance(sub, (int, float))):
            raise ValueError("subsample_n must be a number")
        state = pub.set_viz_config(
            freq_gran=cfg.get("freq_gran"), time_gran=cfg.get("time_gran"),
            agg=cfg.get("agg"), subsample_n=sub)
        return {"ok": True, **state}

    def cmd_start(self, req: dict) -> dict:
        try:
            duration = float(req.get("duration_s", 10.0))
        except (TypeError, ValueError):
            raise ValueError("duration_s must be a number") from None
        if not (0 < duration <= MAX_DURATION_S):
            raise ValueError("duration_s must be in (0, 86400]")
        notes = req.get("notes", "")
        if not isinstance(notes, str) or len(notes.encode()) > MAX_NOTES_BYTES:
            raise ValueError("notes must be a string of at most 16 KiB")
        return self._d._recorder.start(duration, notes)

    def cmd_stop(self, req: dict) -> dict:
        return self._d._recorder.stop()

    def cmd_status(self, req: dict) -> dict:
        return {"ok": True, **self._d._recorder.status()}


def register_handlers(server, dapp) -> ControlHandlers:
    h = ControlHandlers(dapp)
    for cmd in ("caps", "block", "detect", "viz", "start", "stop", "status"):
        server.register(cmd, getattr(h, f"cmd_{cmd}"))
    return h
