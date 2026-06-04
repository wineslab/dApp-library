"""
IQSaver — Python facade over the libiqsaver C++ writer (via SWIG).

This module preserves the original pure-Python IQSaver API so that
existing callers (spectrum_dapp.py, tests/test_iq_saver.py) keep
working unchanged. The actual SigMF serialisation, file rotation,
and annotation buffering now happen in C++.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from iq_saver import iqsaver_native as _native

__author__ = "Andrea Lacava"

# Reserved Python-side keys that should not be forwarded as JSON-encoded
# extra metadata to the C++ writer — they map to dedicated C++ fields.
_RESERVED_GLOBAL_KWARGS = {"sampling_threshold"}

# SigMF datatype string -> bytes per complex sample. The on-disk byte layout the
# writer produces MUST match the declared ``core:datatype`` so that
# ``bytes_on_disk == num_samples * bytes_per_sample`` holds (spear-lake seal-time
# invariant 14). int16-interleaved I/Q is 4 bytes/sample; float32-interleaved is 8.
# Only little-endian formats are supported: the writer streams native-endian numpy
# buffers straight to disk and does NOT byte-swap, so declaring a big-endian
# datatype would mis-describe the on-disk bytes on a little-endian host.
_DTYPE_BYTES_PER_SAMPLE = {
    "ci16_le": 4,
    "cf32_le": 8,
}
# Datatypes whose on-disk element is a raw int16 I/Q pair (no conversion).
_INT16_DTYPES = {"ci16_le"}
# Datatypes whose on-disk element is a float32 I/Q pair (complex64 bytes).
_FLOAT32_DTYPES = {"cf32_le"}


def bytes_per_sample(dtype: str) -> int:
    """Bytes occupied by one complex sample for a SigMF datatype string."""
    try:
        return _DTYPE_BYTES_PER_SAMPLE[dtype]
    except KeyError:
        raise ValueError(
            f"Unsupported SigMF datatype {dtype!r}. "
            f"Supported: {sorted(_DTYPE_BYTES_PER_SAMPLE)}"
        ) from None


def _to_jsonable(value: Any) -> Any:
    """Convert numpy arrays/scalars into JSON-friendly Python primitives."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


class IQSaver:
    """SigMF-compliant IQ-sample recorder (C++ backend via SWIG)."""

    def __init__(
        self,
        base_path: Optional[str] = None,
        center_freq: float = 3.6192e9,
        bandwidth: Optional[float] = None,
        sample_rate: Optional[float] = None,
        annotation_flush_interval: int = 200,
        author: str = "SPEAR dApp",
        description: str = "5G NR Spectrum Sharing IQ Captures",
        hw_info: str = "",
        dtype: str = "ci16_le",
        filename: Optional[str] = None,
        max_samples_per_file: Optional[int] = None,
        rotation_interval: Optional[float] = None,
        **metadata_kwargs: Any,
    ) -> None:
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.center_freq = float(center_freq)
        self.bandwidth = float(bandwidth) if bandwidth else 0.0
        self.sample_rate = float(sample_rate) if sample_rate else self.bandwidth
        # Validates dtype up front and exposes bytes-per-sample for callers/tests.
        self.bytes_per_sample = bytes_per_sample(dtype)
        self.dtype = dtype
        self.annotation_flush_interval = annotation_flush_interval
        self.author = author
        self.description = description
        self.hw_info = hw_info
        self.metadata_kwargs: Dict[str, Any] = dict(metadata_kwargs)

        if max_samples_per_file is not None and max_samples_per_file <= 0:
            raise ValueError("max_samples_per_file must be positive.")
        if rotation_interval is not None and rotation_interval <= 0:
            raise ValueError("rotation_interval must be positive.")

        cfg = _native.IQSaverConfigSwig()
        cfg.base_path = str(self.base_path)
        cfg.center_freq = self.center_freq
        cfg.bandwidth = self.bandwidth
        cfg.sample_rate = self.sample_rate
        cfg.annotation_flush_interval = int(annotation_flush_interval)
        cfg.author = author
        cfg.description = description
        cfg.hw_info = hw_info
        cfg.dtype = dtype
        cfg.filename = filename or ""
        cfg.max_samples_per_file = (
            -1 if max_samples_per_file is None else int(max_samples_per_file)
        )
        cfg.rotation_interval = (
            -1.0 if rotation_interval is None else float(rotation_interval)
        )
        cfg.extension_namespace = "spear"
        cfg.extra_metadata_json = json.dumps(
            {k: _to_jsonable(v) for k, v in metadata_kwargs.items()}
        )

        self._writer = _native.IQSaverWriterSwig(cfg)
        self._closed = False
        # Mirror counters that the Python tests read indirectly through
        # get_recording_info(); the canonical state lives in C++.
        self._session_start_time = time.time()

    # ------------------------------------------------------------------
    # Saving samples
    # ------------------------------------------------------------------

    def save_samples(self, iq_data: np.ndarray,
                     timestamp: Optional[float] = None) -> int:
        if self._closed:
            raise RuntimeError("IQSaver is closed")
        if not isinstance(iq_data, np.ndarray):
            raise TypeError(
                f"iq_data must be a numpy array, got {type(iq_data)!r}")

        # The on-disk byte layout produced here MUST match the declared
        # ``core:datatype`` (self.dtype) — otherwise the SigMF metadata lies about
        # the sample size and spear-lake invariant 14 fails at seal time.
        # We therefore refuse to silently convert across incompatible widths:
        # the data kind and the declared dtype must agree.
        if iq_data.dtype in (np.complex64, np.complex128):
            if self.dtype not in _FLOAT32_DTYPES:
                raise ValueError(
                    f"complex IQ data is written as 8-byte cf32 samples but "
                    f"core:datatype is {self.dtype!r}. Construct IQSaver with "
                    f"dtype='cf32_le' for complex64/complex128 input.")
            # Already-complex64 input only needs a contiguity check (no-op when
            # contiguous); only complex128 needs the narrowing copy. Avoids an
            # unconditional astype copy on the hot path.
            if iq_data.dtype == np.complex64:
                write_data = np.ascontiguousarray(iq_data)
            else:
                write_data = np.ascontiguousarray(iq_data.astype(np.complex64))
            num_samples = len(write_data)
        elif iq_data.dtype == np.int16:
            if len(iq_data) % 2 != 0:
                # Interleaved I/Q must have an even element count. An odd length
                # would write all bytes but count floor(n/2) samples, desyncing
                # iq_sample_count from the bytes on disk (breaking indexing and
                # sample-count rotation).
                raise ValueError(
                    f"int16 IQ data must have an even (interleaved I/Q) length, "
                    f"got {len(iq_data)}.")
            if self.dtype in _INT16_DTYPES:
                # Raw int16 I/Q pairs straight to disk — bytes match ci16_le.
                write_data = np.ascontiguousarray(iq_data)
                num_samples = len(write_data) // 2
            elif self.dtype in _FLOAT32_DTYPES:
                # Explicit, declared int16 -> complex64 conversion — bytes match cf32_le.
                # Build complex64 directly (assigning to .real/.imag casts int16 ->
                # float32 in place) to avoid the complex128 intermediate that
                # `f32 + 1j*f32` would allocate on every write.
                num_samples = len(iq_data) // 2
                write_data = np.empty(num_samples, dtype=np.complex64)
                write_data.real = iq_data[::2]
                write_data.imag = iq_data[1::2]
            else:
                raise ValueError(
                    f"int16 IQ data cannot be written as core:datatype "
                    f"{self.dtype!r}. Use dtype='ci16_le' (raw passthrough) or "
                    f"dtype='cf32_le' (explicit float conversion).")
        else:
            raise ValueError(
                f"Unsupported data type: {iq_data.dtype}. "
                "Use complex64, complex128, or int16")

        ts = -1.0 if timestamp is None else float(timestamp)
        return int(self._writer.save_samples_buf(write_data, int(num_samples), ts))

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    def add_annotation(self,
                       start_sample: Optional[int] = None,
                       label: str = "",
                       comment: str = "",
                       timestamp: Optional[float] = None,
                       **custom_fields: Any) -> bool:
        custom_json = json.dumps(
            {k: _to_jsonable(v) for k, v in custom_fields.items()})
        return bool(self._writer.add_annotation(
            -1 if start_sample is None else int(start_sample),
            label,
            comment,
            -1.0 if timestamp is None else float(timestamp),
            custom_json,
        ))

    def finalize_annotations(self) -> None:
        self._writer.finalize_annotations()

    def update_sample_rate(self,
                           new_sample_rate: float,
                           sampling_threshold: Optional[int] = None) -> None:
        self._writer.update_sample_rate(
            float(new_sample_rate),
            -1 if sampling_threshold is None else int(sampling_threshold),
        )
        if sampling_threshold is not None:
            self.metadata_kwargs["sampling_threshold"] = sampling_threshold

    def add_waveform_description(self,
                                 timestamp: Optional[float] = None,
                                 **kwargs: Any) -> bool:
        fields_json = json.dumps(
            {k: _to_jsonable(v) for k, v in kwargs.items()})
        return bool(self._writer.add_waveform_description(
            -1.0 if timestamp is None else float(timestamp),
            fields_json,
        ))

    # ------------------------------------------------------------------
    # Introspection / lifecycle
    # ------------------------------------------------------------------

    def get_recording_info(self) -> Dict[str, Any]:
        info = json.loads(self._writer.get_recording_info())
        # file_size_bytes is delivered as a JSON object keyed by path;
        # Python tests treat it as a dict, so leave the shape as-is.
        return info

    def close(self) -> None:
        if self._closed:
            return
        self._writer.close()
        self._closed = True

    def __enter__(self) -> "IQSaver":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
