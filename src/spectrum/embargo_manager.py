import numpy as np

class EmbargoManager:
    """
    Manages 5G NR subcarrier blocking decisions using a two-tier detection
    persistence strategy designed for scan-on-scan radar intercept scenarios.

    Tier 1 - Confirmation: A single detection immediately blocks a subcarrier
             and records the time of last detection.
    Tier 2 - Decay: A per-subcarrier smoothed SNR presence score decays
             exponentially between observations. A subcarrier remains blocked
             until both the hold timer has expired AND the smoothed score has
             fallen below the release threshold.

    Parameters
    ----------
    n_bins : int
        Number of FFT bins (subcarriers) to manage.
    hold_time : float
        Minimum time in seconds a subcarrier stays blocked after the last
        detection, regardless of the smoothed score. Should be set to 1-2x
        the longest expected radar scan period (e.g. 20.0 for a 10-second
        phased array).
    alpha_rise : float
        IIR smoothing coefficient applied when a subcarrier is detected
        (0 < alpha < 1). Higher values make the score rise faster. Default 0.7.
    alpha_fall : float
        IIR smoothing coefficient applied when a subcarrier is not detected.
        Lower values make the score decay more slowly. Default 0.05.
    release_threshold : float
        Smoothed score below which a subcarrier is considered clear (after
        hold_time has also expired). Should be well below block_threshold.
        Default 0.1.
    block_threshold : float
        Smoothed score above which a subcarrier is considered active, used as
        a secondary block condition alongside the hold timer. Default 0.5.
    """
    def __init__(
        self,
        n_bins: int,
        hold_time: float,
        alpha_rise: float = 0.7,
        alpha_fall: float = 0.05,
        release_threshold: float = 0.1,
        block_threshold: float = 0.5,
    ):
        self.n_bins = n_bins
        self.hold_time = hold_time
        self.alpha_rise = alpha_rise
        self.alpha_fall = alpha_fall
        self.release_threshold = release_threshold
        self.block_threshold = block_threshold

        # Per-bin smoothed presence score in [0, 1]
        self._score = np.zeros(n_bins, dtype=np.float32)
        # Timestamp of last detection per bin; NaN means never detected
        self._last_detected = np.full(n_bins, np.nan, dtype=np.float64)
        # Hysteresis state: tracks which bins are currently considered blocked
        self._is_blocked = np.zeros(n_bins, dtype=bool)

    def update(self, timestamp: float, detections: np.ndarray) -> None:
        """
        Update internal state with a new observation.

        Parameters
        ----------
        timestamp : float
            Time of this observation in seconds (e.g. time.monotonic()).
        detections : np.ndarray
            Boolean array of length n_bins. True indicates the subcarrier's
            SNR exceeded the detection threshold in this observation window.

        Raises
        ------
        ValueError
            If detections does not have exactly n_bins elements.
        """
        if detections.shape[0] != self.n_bins:
            raise ValueError(
                f"Expected detections of length {self.n_bins}, got {detections.shape[0]}."
            )

        detected = np.asarray(detections, dtype=bool)

        # Update smoothed score with asymmetric IIR
        alpha = np.where(detected, self.alpha_rise, self.alpha_fall)
        target = np.where(detected, 1.0, 0.0)
        self._score = (alpha * target + (1.0 - alpha) * self._score).astype(np.float32)

        # Record timestamp for any newly or re-detected bins
        self._last_detected = np.where(detected, timestamp, self._last_detected)

    def is_blocked(self, timestamp: float) -> np.ndarray:
        """
        Return a boolean array indicating which subcarriers should be blocked.

        A subcarrier is blocked if either:
          - It has been detected and its hold timer has not yet expired, OR
          - Its smoothed presence score is above the block threshold.

        A subcarrier is released only when BOTH conditions are false:
          - The hold timer has expired (or it was never detected), AND
          - The smoothed score has fallen below the release threshold.

        Parameters
        ----------
        timestamp : float
            Current time in seconds (e.g. time.monotonic()).

        Returns
        -------
        np.ndarray
            Boolean array of length n_bins.
        """
        ever_detected = ~np.isnan(self._last_detected)
        # Clamp to zero so a non-monotonic or equal timestamp never makes
        # hold_active fire incorrectly as a "never detected" case.
        time_since = np.where(ever_detected, np.maximum(timestamp - self._last_detected, 0.0), np.inf)

        hold_active = ever_detected & (time_since < self.hold_time)

        # True hysteresis: a bin enters blocked when score >= block_threshold;
        # once blocked it stays blocked until score drops below release_threshold.
        newly_blocked = self._score >= self.block_threshold
        score_active = newly_blocked | (self._is_blocked & (self._score >= self.release_threshold))

        self._is_blocked[:] = hold_active | score_active
        return self._is_blocked.copy()

    def reset(self) -> None:
        """Clear all state, unblocking all subcarriers."""
        self._score[:] = 0.0
        self._last_detected[:] = np.nan
        self._is_blocked[:] = False
