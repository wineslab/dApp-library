#!/usr/bin/env python3
"""
IQSaver - SigMF-compliant IQ sample recorder with time-based file rotation
and deferred annotation support for the SPEAR dApp project.
"""
import time
import numpy as np
from typing import Optional, Dict, List, Any
from pathlib import Path
import sigmf
from sigmf import SigMFFile

__author__ = "Andrea Lacava"


class IQSaver:
    """
    SigMF-compliant IQ sample recorder 
    Features:
    - Single-file recording
    - Timestamp-indexed annotation (can be added post-capture)
    - Semantic waveform description support
    - High-performance write (no compression during capture)
    - Complete SigMF metadata with custom 'spear:' namespace
    - Immediate .sigmf-meta file creation for crash safety
    """
    
    def __init__(self,
                 base_path: str = None,
                 center_freq: float = 3.6192e9,
                 bandwidth: float = None,
                 sample_rate: float = None,
                 annotation_flush_interval: int = 200,
                 author: str = "SPEAR dApp",
                 description: str = "5G NR Spectrum Sharing IQ Captures",
                 hw_info: str = "",
                 dtype: str = "ci16_le",
                 filename: str = None,
                 **metadata_kwargs):
        """
        Initialize IQSaver with SigMF-compliant recording configuration.
        
        Args:
            base_path: Directory for recordings (default: current directory)
            center_freq: Center frequency in Hz
            bandwidth: Signal bandwidth in Hz
            sample_rate: Actual sample rate in Hz (defaults to bandwidth if not provided)
            annotation_flush_interval: Number of annotations before auto-flush (default: 200)
            author: Author/system identifier
            description: Recording session description
            hw_info: Hardware/RU information (e.g., "USRP B210", "RU config")
            dtype: SigMF data type (default: ci16_le for complex int16 little-endian)
            filename: Custom filename (without extension). If None, timestamp-based name is used
            **metadata_kwargs: Additional global metadata fields (stored under spear: namespace)
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Radio configuration
        self.center_freq = center_freq
        self.bandwidth = bandwidth
        self.sample_rate = sample_rate if sample_rate else bandwidth
        self.dtype = dtype
        
        # Recording configuration
        self.annotation_flush_interval = annotation_flush_interval
        self.author = author
        self.description = description
        self.hw_info = hw_info
        self.metadata_kwargs = metadata_kwargs
        
        # Recording state
        self._sigmf: Optional[SigMFFile] = None
        self._file_handle: Optional[object] = None
        self._filename: Optional[str] = None
        self._data_path: Optional[str] = None
        self._sample_count: int = 0
        self._is_initialized: bool = False
        
        # Annotation tracking
        self._annotation_buffer: List[Dict[str, Any]] = []
        self._annotation_count: int = 0
        
        # Session metadata
        self._session_start_time = time.time()
        self._custom_filename = filename
        
    def _initialize_file(self, timestamp: Optional[float] = None) -> None:
        """
        Initialize the SigMF recording file.
        
        Args:
            timestamp: Unix timestamp in seconds (default: current time)
        """
        if self._is_initialized:
            return
            
        # Generate filename with millisecond precision
        if timestamp is None:
            timestamp = time.time()
        
        if self._custom_filename:
            base_filename = self._custom_filename
        else:
            timestamp_ms = int(timestamp * 1000)
            base_filename = f"spectrum_iq_{timestamp_ms}"
        
        self._filename = base_filename
        
        data_path = self.base_path / f"{base_filename}.sigmf-data"
        
        # Open data file for writing
        self._file_handle = open(data_path, 'wb', buffering=1024*1024)  # 1MB buffer
        
        # Create SigMF metadata
        self._sigmf = SigMFFile(
            global_info={
                SigMFFile.DATATYPE_KEY: self.dtype,
                SigMFFile.SAMPLE_RATE_KEY: self.sample_rate,
                SigMFFile.AUTHOR_KEY: self.author,
                SigMFFile.DESCRIPTION_KEY: self.description,
                SigMFFile.VERSION_KEY: "1.0.0",
            }
        )
        
        # Store data file path for later
        self._data_path = str(data_path)
        
        # Add hardware info if provided
        if self.hw_info:
            self._sigmf.set_global_field("core:hw", self.hw_info)
        
        # Add custom SPEAR metadata
        for key, value in self.metadata_kwargs.items():
            if key != 'sampling_threshold':
                self._sigmf.set_global_field(f"spear:{key}", value)
        
        # Add first capture segment with sampling_threshold
        capture_metadata = {
            SigMFFile.FREQUENCY_KEY: self.center_freq,
            SigMFFile.DATETIME_KEY: self._get_iso8601_timestamp(timestamp),
        }
        
        # Add bandwidth if provided
        if self.bandwidth:
            capture_metadata["core:bandwidth"] = self.bandwidth
        
        # Add sampling_threshold to capture metadata if provided
        if 'sampling_threshold' in self.metadata_kwargs:
            capture_metadata["spear:sampling_threshold"] = self.metadata_kwargs['sampling_threshold']
        
        self._sigmf.add_capture(0, metadata=capture_metadata)
        
        # Write initial metadata file immediately (SigMF requires .sigmf-meta alongside .sigmf-data)
        # This ensures the file exists even if the program is interrupted
        self._sigmf.tofile(str(self.base_path / base_filename))
        
        self._is_initialized = True
        
    def _finalize_file(self) -> None:
        """Finalize the recording file and write metadata."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
        
        if self._sigmf is not None and self._filename is not None:
            # Flush any pending annotations
            self._flush_annotations()
            
            # Write final metadata file
            self._sigmf.tofile(str(self.base_path / self._filename))
        
    def _get_iso8601_timestamp(self, timestamp: Optional[float] = None) -> str:
        """Convert Unix timestamp to ISO 8601 format for SigMF with microsecond precision."""
        if timestamp is None:
            timestamp = time.time()
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt.isoformat(timespec='microseconds')
    
    def save_samples(self, iq_data: np.ndarray, timestamp: Optional[float] = None) -> int:
        """
        Save IQ samples to the recording file.
        
        Args:
            iq_data: Complex IQ samples (numpy array of complex type) or int16 interleaved I/Q
            timestamp: Unix timestamp in seconds (default: current time)
            
        Returns:
            sample_index: Sample index for this write (for annotation reference)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Initialize file on first write
        if not self._is_initialized:
            self._initialize_file(timestamp)
        
        # Ensure data is in the correct format for writing
        # SigMF expects complex64 (cf32_le) or complex int16 (ci16_le)
        if iq_data.dtype == np.complex64:
            # Already in correct format for cf32_le
            write_data = iq_data
            num_samples = len(iq_data)
        elif iq_data.dtype == np.complex128:
            # Convert to complex64
            write_data = iq_data.astype(np.complex64)
            num_samples = len(iq_data)
        elif iq_data.dtype == np.int16:
            # Interleaved I/Q int16 - convert to complex for proper sample count
            # But write as-is if dtype is ci16_le
            if self.dtype == "ci16_le":
                write_data = iq_data
                num_samples = len(iq_data) // 2
            else:
                # Convert int16 pairs to complex64
                iq_complex = iq_data[::2].astype(np.float32) + 1j * iq_data[1::2].astype(np.float32)
                write_data = iq_complex.astype(np.complex64)
                num_samples = len(iq_complex)
        else:
            raise ValueError(f"Unsupported data type: {iq_data.dtype}. Use complex64, complex128, or int16")
        
        # Write to file
        write_data.tofile(self._file_handle)
        
        # Update sample count
        sample_index = self._sample_count
        self._sample_count += num_samples
        
        return sample_index
    
    def add_annotation(self,
                      start_sample: Optional[int] = None,
                      label: str = "",
                      comment: str = "",
                      **custom_fields) -> bool:
        """
        Add annotation to capture.
        
        Args:
            start_sample: Sample offset (if None, uses current sample count)
            label: Annotation label (e.g., "prb_control", "interference")
            comment: Human-readable description
            **custom_fields: Custom annotation fields (stored with spear: prefix)
            
        Returns:
            Success boolean
            
        Examples:
            # Annotation at specific sample index
            saver.add_annotation(start_sample=1000, label="prb_blacklist", 
                               prb_list=[76,77,78], reason="interference")
            
            # Annotation at current position
            saver.add_annotation(label="waveform_type",
                               comment="5G NR uplink with PUSCH")
        """
        if not self._is_initialized:
            return False
        
        # Build annotation
        annotation = {
            "sample_start": start_sample if start_sample is not None else self._sample_count,
            "label": label,
        }
        
        if comment:
            annotation["comment"] = comment
        
        # Add custom fields with spear: prefix
        for key, value in custom_fields.items():
            annotation[f"spear:{key}"] = value
        
        # Add to buffer
        self._annotation_buffer.append(annotation)
        self._annotation_count += 1
        
        # Auto-flush if threshold reached
        if self._annotation_count >= self.annotation_flush_interval:
            self.finalize_annotations()
        
        return True
    
    def _flush_annotations(self) -> None:
        """Flush annotations to the metadata file."""
        if self._sigmf is None:
            return
            
        if len(self._annotation_buffer) == 0:
            return
        
        for annotation in self._annotation_buffer:
            start_sample = annotation.get("sample_start", 0)
            
            # Build metadata dict for annotation
            ann_metadata = {}
            
            # Add core fields
            if "label" in annotation:
                ann_metadata[SigMFFile.LABEL_KEY] = annotation["label"]
            if "comment" in annotation:
                ann_metadata[SigMFFile.COMMENT_KEY] = annotation["comment"]
            
            # Add custom spear: fields
            for key, value in annotation.items():
                if key.startswith("spear:") and key not in ["sample_start", "label", "comment"]:
                    ann_metadata[key] = value
            
            # Add annotation without length parameter
            self._sigmf.add_annotation(
                start_sample,  # start index
                metadata=ann_metadata
            )
        
        # Clear buffer
        self._annotation_buffer.clear()
        self._annotation_count = 0
    
    def finalize_annotations(self) -> None:
        """
        Write all pending annotations to the .sigmf-meta file.
        Should be called periodically or at end of session.
        """
        self._flush_annotations()
        
        # Write metadata file to disk after flushing annotations
        if self._sigmf is not None and self._filename is not None:
            self._sigmf.tofile(str(self.base_path / self._filename))
    
    def update_sample_rate(self, new_sample_rate: float, sampling_threshold: int = None) -> None:
        """
        Update the sample rate dynamically during recording.
        
        This method creates a new capture segment to mark the sampling threshold change.
        Note: The global sample_rate is NOT changed - it represents the actual sensing rate.
        Only the spear:sampling_threshold is updated per capture segment.
        Useful when sampling parameters change during a recording session.
        
        Args:
            new_sample_rate: New effective sample rate in Hz (for reference/logging)
            sampling_threshold: New sampling threshold value (creates new capture segment)
            
        Example:
            # Update when sampling threshold changes
            new_rate = 1 / (0.01 * sampling_threshold)
            iq_saver.update_sample_rate(new_rate, sampling_threshold=sampling_threshold)
        """
        # Note: We do NOT update self.sample_rate or the global field
        # The global sample_rate represents the actual sensing rate which is constant
        
        # Create a new capture segment if initialized and sampling_threshold provided
        if self._sigmf is not None and sampling_threshold is not None:
            # Create a new capture segment at the current sample position
            # to mark the change in sampling parameters
            capture_metadata = {
                SigMFFile.FREQUENCY_KEY: self.center_freq,
                SigMFFile.DATETIME_KEY: self._get_iso8601_timestamp(),
            }
            
            if self.bandwidth:
                capture_metadata["core:bandwidth"] = self.bandwidth
            
            capture_metadata["spear:sampling_threshold"] = sampling_threshold
            # Update metadata_kwargs so future references have the new value
            self.metadata_kwargs['sampling_threshold'] = sampling_threshold
            
            # Add new capture segment at current sample position
            self._sigmf.add_capture(self._sample_count, metadata=capture_metadata)
    
    def close(self) -> None:
        """
        Finalize recording and write all metadata.
        Should be called at the end of the recording session.
        """
        # Flush all pending annotations
        self.finalize_annotations()
        
        # Finalize file
        self._finalize_file()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
    
    def __del__(self):
        """Destructor to ensure cleanup on garbage collection."""
        try:
            # Attempt to close any open files if not already closed
            if self._file_handle is not None or self._sigmf is not None:
                self.close()
        except Exception:
            # Silently ignore errors during cleanup
            pass
