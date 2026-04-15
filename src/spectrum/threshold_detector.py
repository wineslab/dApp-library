"""
Threshold detection strategies for the Spectrum Sharing dApp.

Two concrete implementations are provided:

- StaticThresholdDetector:  accumulates N frames, averages per-bin magnitude in dB,
  then compares against a fixed threshold.  Equivalent to the original hardcoded path.

- AdaptiveThresholdDetector:  estimates per-bin noise floor via a running median
  (AdaptiveNoiseFloor) and thresholds the per-bin SNR.  Embargo hysteresis
  (EmbargoManager) prevents PRB flapping.

Both classes share the ThresholdDetector ABC so SpectrumSharingDApp is agnostic to
the active strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from spectrum.adaptive_noise_floor import AdaptiveNoiseFloor
from spectrum.embargo_manager import EmbargoManager


class ThresholdDetector(ABC):
    """Abstract base class for per-bin interference detection.

    All arrays operate in *first-carrier-aligned* space: the caller applies the
    FFT-shift (``np.roll(abs_iq, -first_carrier_offset)``) before calling
    :meth:`update`, so subcarrier index 0 maps to PRB 0.
    """

    @abstractmethod
    def update(
        self,
        abs_iq_shifted: np.ndarray,
        timestamp: float,
    ) -> tuple[bool, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Process one magnitude spectrum and return detection results.

        Parameters
        ----------
        abs_iq_shifted:
            Per-bin magnitude (float32, shape ``(fft_size,)``), FFT-shift
            corrected so that subcarrier 0 is at index 0.
        timestamp:
            Wall-clock seconds (``time.monotonic()``).

        Returns
        -------
        ready:
            ``True`` once enough history has been collected for reliable
            detections.  When ``False``, the other return values should be
            ignored.
        blocked:
            Boolean array (shape ``(fft_size,)``).  ``True`` where interference
            is detected.  Meaningful only when ``ready=True``.
        power_db:
            Per-bin signal power in dB (float32, shape ``(fft_size,)``), or
            ``None`` if not yet available.  Used for visualization.
        noise_floor_db:
            Per-bin estimated noise floor in dB (float32, shape
            ``(fft_size,)``), or ``None`` for static mode / not-yet-ready.
            Used for visualization overlay.
        """

    @property
    @abstractmethod
    def threshold_db(self) -> float:
        """Primary detection threshold in dB (used for logging and annotations)."""


class StaticThresholdDetector(ThresholdDetector):
    """Accumulate *window* frames, compare averaged power-in-dB to a fixed level.

    After each decision the accumulator is reset so the next window starts fresh.
    This replaces the ``abs_iq_av`` / ``control_count`` state that previously
    lived in ``SpectrumSharingDApp``.

    Parameters
    ----------
    threshold_db:
        Detection threshold in dB.
    fft_size:
        Number of FFT bins.
    window:
        Number of frames to accumulate before each decision.  Default 64.
    """

    def __init__(self, threshold_db: float, fft_size: int, window: int = 64) -> None:
        self._threshold_db = float(threshold_db)
        self._fft_size = fft_size
        self._window = window
        # float64 accumulator for numerical stability across many frames
        self._acc = np.zeros(fft_size, dtype=np.float64)
        self._count = 0
        self._last_power_db: np.ndarray | None = None
        # Pre-allocated "not ready" blocked array — avoids a heap allocation on
        # every call while the accumulation window is still filling up.
        self._empty_blocked = np.zeros(fft_size, dtype=bool)

    # ------------------------------------------------------------------
    # ThresholdDetector interface
    # ------------------------------------------------------------------

    def update(
        self,
        abs_iq_shifted: np.ndarray,
        timestamp: float,
    ) -> tuple[bool, np.ndarray, np.ndarray | None, np.ndarray | None]:
        self._acc += abs_iq_shifted
        self._count += 1

        if self._count < self._window:
            return False, self._empty_blocked, None, None

        avg = self._acc / self._window
        self._last_power_db = (
            20.0 * np.log10(np.maximum(avg, 1e-6))
        ).astype(np.float32)

        blocked = self._last_power_db > self._threshold_db

        # Reset for next window
        self._acc[:] = 0.0
        self._count = 0

        return True, blocked, self._last_power_db, None

    @property
    def threshold_db(self) -> float:
        return self._threshold_db

    # ------------------------------------------------------------------
    # Introspection helpers (used by annotations)
    # ------------------------------------------------------------------

    @property
    def window(self) -> int:
        return self._window


class AdaptiveThresholdDetector(ThresholdDetector):
    """Per-bin median noise floor with SNR thresholding and embargo hysteresis.

    Composes :class:`~spectrum.adaptive_noise_floor.AdaptiveNoiseFloor` and
    :class:`~spectrum.embargo_manager.EmbargoManager`; the dApp no longer
    imports those classes directly.

    Parameters
    ----------
    snr_threshold_db:
        Detection threshold in dB above the estimated per-bin noise floor.
    fft_size:
        Number of FFT bins.
    hist_depth:
        Number of frames kept in the :class:`AdaptiveNoiseFloor` circular
        buffer before the first estimate is available.  Default 32.
    embargo_timeout_secs:
        Minimum hold time in seconds after the last detection before a bin
        can be unblocked.  Default 9.9.
    """

    def __init__(
        self,
        snr_threshold_db: float,
        fft_size: int,
        hist_depth: int = 32,
        embargo_timeout_secs: float = 9.9,
    ) -> None:
        self._snr_threshold_db = float(snr_threshold_db)
        self._fft_size = fft_size
        self._noise_floor_est = AdaptiveNoiseFloor(n=fft_size, x=hist_depth)
        self._embargo = EmbargoManager(n_bins=fft_size, hold_time=embargo_timeout_secs)

    # ------------------------------------------------------------------
    # ThresholdDetector interface
    # ------------------------------------------------------------------

    def update(
        self,
        abs_iq_shifted: np.ndarray,
        timestamp: float,
    ) -> tuple[bool, np.ndarray, np.ndarray | None, np.ndarray | None]:
        self._noise_floor_est.update(abs_iq_shifted)

        # Compute per-bin power in dB on every call so the GUI has fresh data
        # even while the buffer is filling up.
        # Clamp to 1.0 (→ 0 dB minimum) rather than 1e-6 (→ −120 dB): we only
        # need the SNR above the per-bin noise floor, so keeping values ≥ 0 dB
        # avoids spurious negative contributions to the SNR calculation.
        # (StaticThresholdDetector uses 1e-6 to preserve the full dynamic range
        # for its fixed absolute threshold; here the threshold is relative.)
        power_db = (
            20.0 * np.log10(np.maximum(abs_iq_shifted, 1.0))
        ).astype(np.float32)

        if not self._noise_floor_est.is_ready:
            return False, np.zeros(self._fft_size, dtype=bool), None, None

        noise_floor = self._noise_floor_est.get_noise_floor()  # float32
        noise_floor_db = (
            20.0 * np.log10(np.maximum(noise_floor, 1.0))
        ).astype(np.float32)

        snr_db = power_db - noise_floor_db
        detections = snr_db >= self._snr_threshold_db

        self._embargo.update(timestamp, detections)
        blocked = self._embargo.is_blocked(timestamp)

        return True, blocked, power_db, noise_floor_db

    @property
    def threshold_db(self) -> float:
        return self._snr_threshold_db

    # ------------------------------------------------------------------
    # Introspection helpers (used by annotations)
    # ------------------------------------------------------------------

    @property
    def hist_depth(self) -> int:
        return self._noise_floor_est.x

    @property
    def embargo_timeout_secs(self) -> float:
        return self._embargo.hold_time
