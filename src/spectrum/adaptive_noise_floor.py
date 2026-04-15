import numpy as np

class AdaptiveNoiseFloor:
  """
  Computes an adaptive noise floor from I/Q frequency-domain data using a
  circular buffer of historical magnitude spectra.

  Parameters
  ----------
  n : int
      Number of frequency bins per I/Q buffer.
  x : int
      Number of historical buffers to retain (depth of the circular buffer).
  """
  def __init__(self, n: int, x: int):
    self.n = n
    self.x = x
    self._buffer = np.zeros((x, n), dtype=np.float32)
    self._head = 0       # next row to write
    self._count = 0      # how many times we've been updated

  def update(self, iq_buffer: np.ndarray) -> None:
    """
    Store a new I/Q frequency-domain buffer in the circular buffer.

    Parameters
    ----------
    iq_buffer : np.ndarray
        1-D real float array of length n containing per-bin magnitudes.

    Raises
    ------
    ValueError
        If iq_buffer does not have exactly n elements.
    """
    if iq_buffer.shape[0] != self.n:
      raise ValueError(f"Expected buffer of length {self.n}, got {iq_buffer.shape[0]}")

    np.copyto(self._buffer[self._head], iq_buffer, casting="unsafe")
    self._head = (self._head + 1) % self.x
    self._count += 1

  def get_noise_floor(self) -> np.ndarray | None:
    """
    Compute the per-bin median magnitude across the history buffer.

    Returns None until the circular buffer has been filled at least once.

    Returns
    -------
    np.ndarray of shape (n,) and dtype float32, or None.
    """
    if self._count < self.x:
      return None

    return np.median(self._buffer, axis=0).astype(np.float32)

  def reset(self) -> None:
    """Reset the circular buffer, discarding all history."""
    self._buffer[:] = 0.0
    self._head = 0
    self._count = 0

  @property
  def is_ready(self) -> bool:
    """True once the circular buffer has been filled at least once."""
    return self._count >= self.x
