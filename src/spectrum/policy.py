"""Policy interface for the spectrum-sharing dApp.

A policy decides which PRBs the gNB must stop using. The dApp feeds it one
:class:`Observation` per detector run and sends the PRB blacklist it returns.
Everything else (E3 connection, subscriptions, DC-leakage guard, control
reconciliation, which sends a control only when the set changed) stays in
:class:`spectrum.spectrum_dapp.SpectrumSharingDApp`.

Write a policy by subclassing :class:`Policy`::

    class MyPolicy(Policy):
        def update(self, obs):
            hot = np.where(obs.prb_power_db > 50)[0]
            return set(hot.tolist())

and run it with ``python3 -m examples.spectrum_dapp --policy my_module:MyPolicy ...``.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

import numpy as np

from spectrum.threshold_detector import ThresholdDetector

SC_PER_PRB = 12


@dataclass(frozen=True)
class Observation:
    """One spectrum measurement.

    Attributes
    ----------
    power_db:
        Per-subcarrier magnitude in dB (``20*log10`` of the mean magnitude over the
        sensed symbols, floored at 0 dB), shape ``(num_prbs * 12,)``. The reference is
        the gNB's rescaled FFT counts, not dBm.
    prb_power_db:
        Per-PRB level in dB, shape ``(num_prbs,)``: ``20*log10`` of the mean magnitude
        of the measured subcarriers of the PRB (``nan`` when none were measured).
    valid:
        Boolean array, shape ``(num_prbs * 12,)``: ``True`` where the subcarrier was
        actually measured in this observation (the rest of the slot was scheduled to UEs).
    t:
        Monotonic time of the observation in seconds.
    """

    power_db: np.ndarray
    prb_power_db: np.ndarray
    valid: np.ndarray
    t: float

    @property
    def num_prbs(self) -> int:
        return int(self.prb_power_db.shape[0])


class Policy(ABC):
    """Base class of a PRB-blocking policy.

    The dApp calls :meth:`update` once per observation (every few milliseconds, about
    6.7 ms nominal, depending on how fast the gNB runs). It returns the set of PRB
    indices to block; PRBs that are not returned are released. State between calls lives on ``self``;
    ``self.history`` holds the last ``history_len`` observations, the current one last.
    """

    def __init__(self, history_len: int = 64) -> None:
        self.history: deque[Observation] = deque(maxlen=history_len)

    def reset(self) -> None:
        """Forget all state. Called once before the first observation."""
        self.history.clear()

    @abstractmethod
    def update(self, obs: Observation) -> set[int]:
        """Return the set of PRB indices to block after seeing ``obs``."""

    def observe(self, obs: Observation) -> set[int]:
        """Called by the dApp: records the observation, runs :meth:`update`, checks the result."""
        self.history.append(obs)
        out = self.update(obs)
        if out is None:
            return set()
        blocked = {int(p) for p in out}
        bad = [p for p in blocked if not 0 <= p < obs.num_prbs]
        if bad:
            raise ValueError(f"policy returned PRBs outside 0..{obs.num_prbs - 1}: {sorted(bad)}")
        return blocked


class PolicyDetector(ThresholdDetector):
    """Runs a :class:`Policy` where the dApp expects a threshold detector."""

    def __init__(self, policy: Policy, fft_size: int, num_prbs: int) -> None:
        self.policy = policy
        self._fft_size = fft_size
        self._num_prbs = num_prbs
        self._n_sc = num_prbs * SC_PER_PRB
        self._reset_done = False

    @property
    def threshold_db(self) -> float:
        return float(getattr(self.policy, "threshold_db", 0.0))

    def update(self, abs_iq_shifted, timestamp, valid_mask=None):
        if not self._reset_done:
            self.policy.reset()
            self._reset_done = True
        n = self._n_sc
        mag = np.asarray(abs_iq_shifted[:n], dtype=np.float32)
        valid = np.ones(n, dtype=bool) if valid_mask is None else np.asarray(valid_mask[:n], dtype=bool)
        power_db = 20.0 * np.log10(np.maximum(mag, 1.0))

        mag_prb = np.where(valid, mag, 0.0).reshape(self._num_prbs, SC_PER_PRB).sum(axis=1)
        cnt = valid.reshape(self._num_prbs, SC_PER_PRB).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_prb = np.where(cnt > 0, mag_prb / np.maximum(cnt, 1), np.nan)
            prb_power_db = 20.0 * np.log10(np.maximum(mean_prb, 1.0))
        prb_power_db = np.where(cnt > 0, prb_power_db, np.nan).astype(np.float32)

        obs = Observation(power_db=power_db, prb_power_db=prb_power_db, valid=valid, t=float(timestamp))
        blocked_prbs = self.policy.observe(obs)

        blocked = np.zeros(self._fft_size, dtype=bool)
        for p in blocked_prbs:
            blocked[p * SC_PER_PRB:(p + 1) * SC_PER_PRB] = True
        full_power = np.zeros(self._fft_size, dtype=np.float32)
        full_power[:n] = power_db
        return True, blocked, full_power, None


class BaselinePolicy(Policy):
    """Reference solution: block PRBs well above the noise, release after a hold time.

    The noise level is the median PRB level of the observation, so no calibration is
    needed as long as the incumbent covers fewer than half of the PRBs.
    """

    def __init__(self, margin_db: float = 12.0, hold_s: float = 1.0, history_len: int = 64) -> None:
        super().__init__(history_len)
        self.margin_db = margin_db
        self.hold_s = hold_s
        self._last_hot: dict[int, float] = {}

    def reset(self) -> None:
        super().reset()
        self._last_hot.clear()

    def update(self, obs: Observation) -> set[int]:
        level = obs.prb_power_db
        floor = np.nanmedian(level)
        for prb in np.where(level > floor + self.margin_db)[0]:
            self._last_hot[int(prb)] = obs.t
        self._last_hot = {p: t for p, t in self._last_hot.items() if obs.t - t < self.hold_s}
        return set(self._last_hot)


def load_policy(spec: str, **kwargs) -> Policy:
    """Instantiate a policy from ``module:Class`` (the module must be importable)."""
    mod_name, _, cls_name = spec.partition(":")
    if not mod_name or not cls_name:
        raise ValueError(f"--policy must look like 'module:Class', got {spec!r}")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    policy = cls(**kwargs)
    if not isinstance(policy, Policy):
        raise TypeError(f"{spec} is not a spectrum.policy.Policy")
    return policy
