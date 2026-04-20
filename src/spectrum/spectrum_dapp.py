#!/usr/bin/env python3
"""
dApp for Spectrum Sharing
"""

__author__ = "Andrea Lacava"

import math
import threading
import queue
import time
import os
import json
from typing import override
import numpy as np
import asn1tools
import jsonschema
# np.set_printoptions(threshold=sys.maxsize)

from dapp.dapp import DApp
from e3interface.e3_encoder import JsonE3Encoder
from e3interface.e3_logging import dapp_logger, LOG_DIR
from spectrum.threshold_detector import (
    ThresholdDetector,
    StaticThresholdDetector,
    AdaptiveThresholdDetector,
)


def compute_fft_size(num_prbs: int, e_sampling: bool = False) -> int:
    """Compute the FFT size for a given PRB count and sampling mode.

    Matches the OAI gNB logic: next power-of-2 of the OFDM symbol size,
    optionally scaled to 3/4 for USRP -E flag (3/4 sampling).
    """
    ofdm_symbol_size = num_prbs * 12  # 12 subcarriers per PRB (LTE/NR standard)
    fft_size = 2 ** math.ceil(math.log2(ofdm_symbol_size))
    if e_sampling:
        fft_size = int(fft_size * 3 / 4)
    return fft_size


class SpectrumSharingDApp(DApp):

    _SPECTRUM_JSON_BINARY_FIELDS = {
        "Spectrum-IQDataIndication": ["iqSamples"],
        "Spectrum-PRBBlacklistControl": ["blacklistedPRBs"],
        "Spectrum-PRBBlacklistReport": ["blacklistedPRBs"]
    }

    # dApp metadata
    DAPP_NAME = "SpectrumSharingDApp"
    DAPP_VERSION = "1.0.0"
    VENDOR = "WinesLab"
    E3AP_PROTOCOL_VERSION = "1.0.0"

    # IDs of interest for this dApp (IQs + PRB Control)
    RAN_FUNCTION_ID = 1
    TELEMETRY_ID = [1]
    CONTROL_ID = [1]

    # PRBs around the DC subcarrier that exhibit leakage artefacts and must
    # be excluded from both reports and control messages.
    DC_LEAKAGE_PRB_LOW = 50
    DC_LEAKAGE_PRB_HIGH = 55

    ###  Default Configuration ###
    # gNB runs with BW = ~40 MHz
    # Center frequency = 3.6192 GHz
    # OAI sampling frequency 46.08e6 # not useful for visualization
    # No SRS for now, hence in gNB config file set do_SRS = 0
    # gNB->frame_parms.ofdm_symbol_size = 1536 # FFT size for USRP with -E flag (3/4 sampling)
    # gNB->frame_parms.ofdm_symbol_size = 2048 # FFT size for RUs without -E flag
    # Noise floor threshold needs to be calibrated according to the RU
    # IQ symbols are captured, passed through the detector strategy, and
    # the resulting PRB thresholding is sent back to the gNB

    def __init__(self, dapp_name: str = DAPP_NAME, dapp_version: str = DAPP_VERSION,
                 vendor: str = VENDOR, e3ap_protocol_version: str = E3AP_PROTOCOL_VERSION,
                 link: str = 'posix', transport: str = 'ipc',
                 detector: ThresholdDetector | None = None,
                 save_iqs: bool = False, control: bool = False,
                 center_freq: float = 3.6192e9, num_prbs: int = 106,
                 num_subcarrier_spacing: int = 30,
                 e_sampling: bool = False, encoding_method: str = "asn1",
                 sampling_threshold: int = 5, **kwargs):
        super().__init__(dapp_name=dapp_name, dapp_version=dapp_version, vendor=vendor,
                         e3ap_protocol_version=e3ap_protocol_version, link=link,
                         transport=transport, encoding_method=encoding_method, **kwargs)

        # Initialize spectrum encoder based on encoding method
        self._init_spectrum_encoder()

        # Custom control logic callback
        self._sampling_threshold_control_callback = None

        # gNB radio configuration
        self.num_consecutive_subcarriers_for_prb: int = 12  # Fixed by LTE/NR standard
        self.num_prbs = num_prbs
        self.num_subcarrier_spacing = num_subcarrier_spacing  # subcarrier spacing in kHz
        self.ofdm_symbol_size = num_prbs * self.num_consecutive_subcarriers_for_prb
        self.bw = (self.ofdm_symbol_size * self.num_subcarrier_spacing * 1e3)  # Hz
        self.center_freq = center_freq
        self.fft_size = compute_fft_size(num_prbs, e_sampling)
        self.first_carrier_offset = self.fft_size - (self.ofdm_symbol_size // 2)

        # dApp configuration
        # First 75 PRBs contain BWP/PRACH channels that must not be blacklisted.
        self.prb_thrs = 75
        self.save_iqs = save_iqs
        self.sampling_threshold = sampling_threshold

        # Detection strategy
        if detector is None:
            raise ValueError("A ThresholdDetector instance must be provided via the 'detector' parameter")
        self._detector = detector
        dapp_logger.info(f"Detector: {type(self._detector).__name__}, threshold: {self._detector.threshold_db} dB")

        # Pre-allocated float32 I/Q buffers — reused every indication, no complex128 intermediate.
        # _mag_buf holds the per-bin magnitude so _I_buf and _Q_buf remain intact after hypot.
        self._I_buf = np.empty(self.fft_size, dtype=np.float32)
        self._Q_buf = np.empty(self.fft_size, dtype=np.float32)
        self._mag_buf = np.empty(self.fft_size, dtype=np.float32)
        # Pre-allocated buffer for the FFT-shifted magnitude — avoids the np.roll allocation
        self._abs_shifted_buf = np.empty(self.fft_size, dtype=np.float32)

        # IQ recording
        if self.save_iqs:
            from iq_saver.iq_saver import IQSaver
            # This sample_rate needs to be dicussed
            # It might make sense to calculate effective sample rate based on sampling_threshold
            # Each capture is: 10ms * sampling_threshold
            sample_rate = 100  # Hz: sensing done once every 10 ms
            dapp_logger.info(f"Sensing sample rate: {sample_rate:.2f} Hz (each sensing is 10 ms)")
            self.iq_saver = IQSaver(
                base_path=LOG_DIR,
                center_freq=self.center_freq,
                bandwidth=self.bw,
                sample_rate=sample_rate,
                annotation_flush_interval=10,
                hw_info=f"FFT:{self.fft_size}, PRBs:{self.num_prbs}, E-sampling:{e_sampling}",
                description=(
                    f"5G NR Uplink capture from SpectrumSharing dApp"
                    f" - RAN Function {self.RAN_FUNCTION_ID}"
                    f" - detector: {type(self._detector).__name__}"
                    f" - threshold: {self._detector.threshold_db} dB"
                ),
                fft_size=self.fft_size,
                dtype="ci16_le",
                num_prbs=self.num_prbs,
                subcarrier_spacing_khz=self.num_subcarrier_spacing,
                sampling_threshold=self.sampling_threshold,
                max_samples_per_file=2000,
                average_over_frames=(
                    self._detector.window
                    if isinstance(self._detector, StaticThresholdDetector) else None
                )
            )

        self.control = control
        dapp_logger.info(f"Control is {'not ' if not self.control else ''}active")

        self.energyGui = kwargs.get('energyGui', False)
        self.iqPlotterGui = kwargs.get('iqPlotterGui', False)
        self.dashboard = kwargs.get('dashboard', False)
        if self.save_iqs:
            self._ground_truth_label = kwargs.get('ground_truth', "")
            self._ground_truth_lock = threading.Lock()

        # Thread-safe sample_idx for IQ saver annotation cross-reference
        self._sample_idx_lock = threading.Lock()
        self.sample_idx = None

        if self.energyGui:
            from visualization.energy import EnergyPlotter
            self.sig_queue = queue.Queue()
            self.energyPlotter = EnergyPlotter(
                self.fft_size, bw=self.bw, center_freq=self.center_freq
            )

        if self.iqPlotterGui:
            from visualization.iq import IQPlotter
            self.iq_queue = queue.Queue()
            self.iqPlotter = IQPlotter(
                buffer_size=500, fft_size=self.fft_size,
                bw=self.bw, center_freq=self.center_freq,
            )

        if self.dashboard:
            from visualization.dashboard import Dashboard
            self.demo_queue = queue.Queue()
            classifier = kwargs.get('classifier', None)
            self.demo = Dashboard(
                buffer_size=100, ofdm_symbol_size=self.ofdm_symbol_size,
                first_carrier_offset=self.first_carrier_offset,
                bw=self.bw, center_freq=self.center_freq, num_prbs=num_prbs,
                prb_protected_below=self.prb_thrs,
                classifier=classifier,
                adaptiveThreshold=isinstance(self._detector, AdaptiveThresholdDetector),
                control=self.control,
                label_callback=self.set_ground_truth_label if self.save_iqs else None,
                initial_label=self._ground_truth_label if self.save_iqs else "",
                show_controls=kwargs.get('show_controls', False),
            )

    def _init_spectrum_encoder(self):
        """Initialize the spectrum encoder based on the encoding method"""
        match self.encoding_method:
            case "asn1":
                asn_file_path = os.path.join(os.path.dirname(__file__), "defs", "e3sm_spectrum.asn")
                self.spectrum_encoder = asn1tools.compile_files(asn_file_path, codec="per")
            case "json":
                json_schema_path = os.path.join(os.path.dirname(__file__), "defs", "e3sm_spectrum.json")
                with open(json_schema_path, 'r') as f:
                    self.spectrum_schema = json.load(f)
                self.spectrum_validator_cls = jsonschema.validators.validator_for(self.spectrum_schema)
                self.spectrum_validator_cls.check_schema(self.spectrum_schema)
                self.spectrum_registry = self._build_spectrum_registry()
                self.spectrum_encoder = "json"
                dapp_logger.info("Spectrum JSON encoder initialized")
            case _:
                raise ValueError(f"Unsupported encoding method: {self.encoding_method}")

    def _build_spectrum_registry(self):
        import referencing
        import referencing.jsonschema
        resource = referencing.Resource.from_contents(
            self.spectrum_schema,
            default_specification=referencing.jsonschema.DRAFT202012,
        )
        base_uri = self.spectrum_schema.get("$id", "urn:e3sm-spectrum-schema")
        return referencing.Registry().with_resource(base_uri, resource)

    def _validate_spectrum_message(self, message_type: str, data: dict) -> None:
        """Validate a spectrum message dict against its schema definition"""
        if message_type not in self.spectrum_schema.get("$defs", {}):
            return
        validator = self.spectrum_validator_cls(
            self.spectrum_schema["$defs"][message_type],
            registry=self.spectrum_registry,
        )
        validator.validate(data)

    def set_sampling_threshold_control_logic(self, callback):
        """Set a custom control logic callback.

        The callback is invoked after each detection with the signature::

            callback(prb_blacklist: np.ndarray, power_db: np.ndarray) -> tuple[bool, int]

        Args:
            callback: A callable that takes:
                - prb_blacklist (np.ndarray): Blacklisted PRB indices.
                - power_db (np.ndarray): Per-bin signal power in dB, shape
                  ``(fft_size,)``, first-carrier-aligned.  Available in both
                  static and adaptive mode.

                Returns:
                - update_sampling (bool): Whether to update the sampling threshold.
                - sampling_threshold (int): New sampling threshold value (0–100).

        Example::

            def my_logic(prb_blacklist, power_db):
                update = len(prb_blacklist) > 10
                return update, 10 if update else 5

            dapp.set_sampling_threshold_control_logic(my_logic)
        """
        if callback is not None and not callable(callback):
            raise ValueError("Callback must be callable")
        self._sampling_threshold_control_callback = callback
        dapp_logger.info(f"Custom control logic callback {'set' if callback else 'removed'}")

    def set_ground_truth_label(self, label: str):
        with self._ground_truth_lock:
            self._ground_truth_label = label
        dapp_logger.info(f"Ground truth label set to: {label!r}")
        if self.dashboard:
            self.demo.emit_label(label)

    def create_prb_blacklist_control(self, blacklisted_prbs: list[int],
                                     update_sampling: bool = False,
                                     validity_period: int = None) -> bytes:
        """Create a PRB blacklist control message
        
        Args:
            blacklisted_prbs: List of PRB indices
            update_sampling: Whether to include the updated sampling threshold
            validity_period: How long this blacklist is valid in seconds (optional)
            
        Returns:
            Encoded bytes for E3-DAppControlAction.actionData
        """
        control_data = {"blacklistedPRBs": blacklisted_prbs}
        if update_sampling:
            control_data["samplingThreshold"] = self.sampling_threshold
        if validity_period is not None:
            control_data["validityPeriod"] = validity_period
        dapp_logger.debug(control_data)
        return self._encode_spectrum_message("Spectrum-PRBBlacklistControl", control_data)

    def create_prb_blacklist_report(self, blacklisted_prbs: list[int]) -> bytes:
        """Create a PRB blacklist report message

        Args:
            blacklisted_prbs: List of PRB indices
            
        Returns:
            Encoded bytes for E3-DAppReport.reportData
        """
        report_data = {"blacklistedPRBs": blacklisted_prbs}
        dapp_logger.debug(report_data)
        return self._encode_spectrum_message("Spectrum-PRBBlacklistReport", report_data)

    def _encode_spectrum_message(self, message_type: str, data: dict) -> bytes:
        """Encode a spectrum message using the configured encoding method
        
        Args:
            message_type: The spectrum message type to encode
            data: The data dictionary to encode
            
        Returns:
            Encoded bytes
        """
        if self.encoding_method == "asn1":
            if self.spectrum_encoder is None:
                raise RuntimeError("ASN.1 encoder not initialized")
            return self.spectrum_encoder.encode(message_type, data)
        elif self.encoding_method == "json":
            # Convert bytes fields to hex strings for JSON
            json_data = JsonE3Encoder.prepare_data_for_json_encode(
                message_type, data.copy(), self._SPECTRUM_JSON_BINARY_FIELDS
            )
            self._validate_spectrum_message(message_type, json_data)
            return json.dumps(json_data).encode('utf-8')
        else:
            raise ValueError(f"Unsupported encoding method: {self.encoding_method}")

    def decode_iq_data_indication(self, data: bytes) -> dict:
        """Decode an IQ data indication message
        
        Args:
            data: Encoded bytes from E3-IndicationMessage.protocolData
            
        Returns:
            Dictionary with keys: ``iqSamples`` (bytes), ``sampleCount`` (int),
            ``timestamp`` (int, optional).
        """
        return self._decode_spectrum_message("Spectrum-IQDataIndication", data)

    def decode_config_control(self, data: bytes) -> dict:
        """Decode a spectrum configuration control message
        
        Args:
            data: Encoded bytes from E3-DAppControlAction.actionData
            
        Returns:
            Dictionary containing the decoded control data
        """
        return self._decode_spectrum_message("Spectrum-ConfigControl", data)

    @override
    def _decode_ran_function_data(self, data_bytes: bytes) -> dict | None:
        """Decode the ``ranFunctionData`` attached to a SetupResponse entry."""
        return self._decode_spectrum_message("Spectrum-RanFunctionData", data_bytes)

    def _decode_spectrum_message(self, message_type: str, data: bytes) -> dict:
        """Decode a spectrum message using the configured encoding method
        
        Args:
            message_type: The spectrum message type to decode
            data: The encoded bytes to decode
            
        Returns:
            Decoded data dictionary
        """
        if self.encoding_method == "asn1":
            if self.spectrum_encoder is None:
                raise RuntimeError("ASN.1 encoder not initialized")
            return self.spectrum_encoder.decode(message_type, data)
        elif self.encoding_method == "json":
            decoded_data = json.loads(data.decode('utf-8'))
            self._validate_spectrum_message(message_type, decoded_data)
            return JsonE3Encoder.prepare_data_from_json_decode(
                message_type, decoded_data, self._SPECTRUM_JSON_BINARY_FIELDS
            )
        else:
            raise ValueError(f"Unsupported encoding method: {self.encoding_method}")

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _compute_prb_blacklist(self, detected_prbs: np.ndarray) -> np.ndarray:
        """Compute the PRB subset sent in the control blacklist message.

        Applies two exclusions on the already DC-leakage-stripped detected_prbs:
        - BWP/PRACH protected zone: PRBs 0 through prb_thrs (inclusive)
        - Guard band: PRBs num_prbs and above

        DC leakage artefacts are stripped upstream in ``_handle_indication``
        before this method is called, so they are not filtered again here.

        Args:
            detected_prbs: uint16 array of detected PRB indices, already
                stripped of DC leakage artefacts.

        Returns:
            uint16 array of PRB indices to include in the control message.
        """
        return detected_prbs[
            (detected_prbs > self.prb_thrs) & (detected_prbs < self.num_prbs)
        ]

    def _build_annotation_fields(
        self,
        all_prbs: np.ndarray,
        prb_blk_list: np.ndarray,
        power_db: np.ndarray,
        noise_floor_db: np.ndarray | None,
        timestamp,
        comment: str,
        control_fired: bool,
    ) -> dict:
        """Build the full annotation metadata dict for an IQSaver annotation.

        Returns a dict suitable for keyword-unpacking into ``add_annotation``.
        Per-PRB summaries are included so individual recordings are fully
        self-documenting without requiring re-analysis.

        all_prbs: every PRB above the detection threshold (unfiltered, including
                  protected-zone PRBs).  prb_blk_list: the filtered blacklist
                  (eligible zone only, empty when control is off).
        """
        sc = self.num_consecutive_subcarriers_for_prb

        # All detected peaks (unfiltered) — always recorded for analysis
        all_sc_starts = all_prbs * sc
        all_power_per_prb = [float(power_db[s: s + sc].mean()) for s in all_sc_starts]
        if isinstance(self._detector, StaticThresholdDetector):
            all_thresh_per_prb = [self._detector.threshold_db] * len(all_prbs)
        elif noise_floor_db is not None:
            all_thresh_per_prb = [float(noise_floor_db[s: s + sc].mean()) for s in all_sc_starts]
        else:
            all_thresh_per_prb = [self._detector.threshold_db] * len(all_prbs)

        label = "prb_control" if control_fired else "prb_detection"

        base = dict(
            label=label,
            timestamp=timestamp,
            comment=comment,
            all_detected_prbs=all_prbs.tolist(),
            all_power_db_per_prb=all_power_per_prb,
            all_threshold_db_per_prb=all_thresh_per_prb,
            # prb_blacklist only populated when a control message was actually sent;
            # otherwise the eligible-zone filtering is irrelevant to the recording.
            prb_blacklist=prb_blk_list.tolist() if control_fired else [],
            noise_threshold=self._detector.threshold_db,
            power_db_max=float(power_db.max()),
            control_action="blacklist" if control_fired else "none",
        )

        if isinstance(self._detector, StaticThresholdDetector):
            base.update(
                detector="static",
                window_frames=self._detector.window,
            )
        elif isinstance(self._detector, AdaptiveThresholdDetector) and noise_floor_db is not None:
            blk_sc_starts = prb_blk_list * sc
            nf_per_prb = [float(noise_floor_db[s: s + sc].mean()) for s in blk_sc_starts]
            snr_per_prb = [float(power_db[s: s + sc].mean() - noise_floor_db[s: s + sc].mean())
                           for s in blk_sc_starts]
            base.update(
                detector="adaptive",
                hist_depth=self._detector.hist_depth,
                embargo_timeout_secs=self._detector.embargo_timeout_secs,
                noise_floor_per_prb=nf_per_prb,
                snr_db_per_prb=snr_per_prb,
            )

        return base

    @override
    def _handle_xapp_control(self, dapp_identifier: int, data: bytes):
        dapp_logger.info(f'Triggered control callback for dApp {dapp_identifier}')

        decoded = self._decode_spectrum_message("Spectrum-PRBBlockedControl", data)
        prb_blk_list = decoded["blockedPRBs"]
        dapp_logger.info(f"xApp control: blockedPRBs={prb_blk_list}")

        control_payload = self.create_prb_blacklist_control(blacklisted_prbs=prb_blk_list)
        self.e3_interface.schedule_control(
            dappId=self.dapp_id,
            ranFunctionId=self.RAN_FUNCTION_ID,
            controlId=self.CONTROL_ID[0],
            actionData=control_payload,
        )
        dapp_logger.info(f"Sending Control to RAN: blockedPRBs={prb_blk_list}")

        if self.save_iqs:
            if hasattr(self, '_ground_truth_label'):
                with self._ground_truth_lock:
                    ground_truth_label = self._ground_truth_label
            with self._sample_idx_lock:
                if self.sample_idx is not None:
                    self.iq_saver.add_annotation(
                        start_sample=self.sample_idx,
                        label="prb_control",
                        comment=f"Blacklisted {len(prb_blk_list)} PRBs upon message from xApp",
                        timestamp=time.time(),
                        prb_blacklist=prb_blk_list,
                        noise_threshold=self._detector.threshold_db,
                        control_action="blacklist",
                        detector=type(self._detector).__name__,
                        **({"ground_truth_label": ground_truth_label} if hasattr(self, '_ground_truth_label') else {}),
                    )
                    dapp_logger.info("Annotation added")

    @override
    def _handle_indication(self, dapp_identifier: int, data: bytes):
        indication_message = self.decode_iq_data_indication(data)
        iqs_raw = indication_message["iqSamples"]
        sample_count = indication_message["sampleCount"]
        timestamp = indication_message.get("timestamp", None)
        now = time.monotonic()
        dapp_logger.debug(
            f"Received {sample_count} samples, timestamp={timestamp}, monotonic={now:.3f}"
        )

        # Zero-copy int16 view into the received bytes
        iq_arr = np.frombuffer(iqs_raw, dtype=np.int16)
        dapp_logger.debug(f"iq_arr shape: {iq_arr.shape}")

        # Feed raw int16 to optional plotters before any conversion
        if self.iqPlotterGui:
            self.iq_queue.put(iq_arr)
        if self.dashboard:
            self.demo_queue.put(("iq_data", iq_arr))

        # Save raw IQ immediately (int16 view is valid while iqs_raw is alive)
        if self.save_iqs:
            with self._sample_idx_lock:
                self.sample_idx = self.iq_saver.save_samples(iq_arr, timestamp=timestamp)

        # Magnitude spectrum: float32 via pre-allocated buffers, no complex128 intermediate.
        # _mag_buf receives the result so _I_buf and _Q_buf are left intact.
        np.copyto(self._I_buf, iq_arr[::2], casting="unsafe")    # int16 → float32
        np.copyto(self._Q_buf, iq_arr[1::2], casting="unsafe")
        np.hypot(self._I_buf, self._Q_buf, out=self._mag_buf)

        # Single FFT-shift: align first subcarrier to index 0 for PRB mapping.
        # Equivalent to np.roll(self._mag_buf, -k) but writes into a pre-allocated
        # buffer to avoid a heap allocation on every indication.
        k = self.first_carrier_offset
        self._abs_shifted_buf[:self.fft_size - k] = self._mag_buf[k:]
        self._abs_shifted_buf[self.fft_size - k:] = self._mag_buf[:k]
        abs_shifted = self._abs_shifted_buf

        # Detection (strategy-agnostic)
        ready, blocked, power_db, noise_floor_db = self._detector.update(abs_shifted, now)

        if not ready:
            return

        # Visualization
        if self.energyGui:
            self.sig_queue.put((power_db, noise_floor_db))
        if self.dashboard and noise_floor_db is not None:
            self.demo_queue.put(("adaptive_noise_floor", noise_floor_db))

        # All PRBs where the detector flagged at least one subcarrier — reported and annotated as-is.
        sc = self.num_consecutive_subcarriers_for_prb
        detected_prbs = np.unique(
            np.where(blocked)[0] // sc
        ).astype(np.uint16)

        dapp_logger.info(
            f"Detected PRBs ({detected_prbs.size}): {detected_prbs.tolist()} | "
            f"detector={type(self._detector).__name__} | "
            f"power_db_max={power_db.max():.1f} dB"
        )

        # Optional sampling-threshold control callback
        update_sampling = False
        if self._sampling_threshold_control_callback is not None:
            try:
                update_sampling, new_sampling_threshold = (
                    self._sampling_threshold_control_callback(detected_prbs, power_db)
                )
                if update_sampling:
                    self.sampling_threshold = new_sampling_threshold
                    dapp_logger.info(
                        f"Custom logic updated sampling threshold to {self.sampling_threshold}"
                    )
                    if self.save_iqs:
                        new_sample_rate = 1 / (0.01 * self.sampling_threshold)
                        self.iq_saver.update_sample_rate(
                            new_sample_rate,
                            sampling_threshold=self.sampling_threshold,
                        )
                        dapp_logger.info(
                            f"Updated IQ saver sample rate to {new_sample_rate:.2f} Hz"
                            f" (sampling_threshold={self.sampling_threshold})"
                        )
            except Exception:
                dapp_logger.exception("Error in custom control callback")
                update_sampling = False

        # Strip DC leakage artefacts from everything: report, annotation, and control.
        dc_low, dc_high = self.DC_LEAKAGE_PRB_LOW, self.DC_LEAKAGE_PRB_HIGH
        detected_prbs = detected_prbs[(detected_prbs < dc_low) | (detected_prbs > dc_high)]
        reported_prbs = detected_prbs
        report_payload = self.create_prb_blacklist_report(
            blacklisted_prbs=reported_prbs.astype(int).tolist()
        )
        self.e3_interface.schedule_report(
            dappId=self.dapp_id,
            ranFunctionId=self.RAN_FUNCTION_ID,
            reportData=report_payload,
        )

        if self.control:
            # Filter out BWP/PRACH + guard band — only for the control message to the gNB.
            prb_blk_list = self._compute_prb_blacklist(detected_prbs)
            dapp_logger.info(f"Control blacklist ({prb_blk_list.size}): {prb_blk_list.tolist()}")
            control_payload = self.create_prb_blacklist_control(
                blacklisted_prbs=prb_blk_list.astype(int).tolist(), update_sampling=update_sampling
            )
            self.e3_interface.schedule_control(
                dappId=self.dapp_id,
                ranFunctionId=self.RAN_FUNCTION_ID,
                controlId=self.CONTROL_ID[0],
                actionData=control_payload,
            )
        else:
            prb_blk_list = np.empty(0, dtype=np.uint16)

        # Annotate the IQ recording with full detection metadata (always, not only when control).
        # Build the annotation dict before acquiring the lock — _build_annotation_fields
        # only reads local variables and immutable detector properties, so it is safe
        # to run outside the critical section, keeping the lock hold-time minimal.
        if self.save_iqs:
            ann = self._build_annotation_fields(
                all_prbs=detected_prbs,
                prb_blk_list=prb_blk_list,
                power_db=power_db,
                noise_floor_db=noise_floor_db,
                timestamp=timestamp,
                comment=(
                    f"Blacklisted {prb_blk_list.size} PRBs due to interference"
                    if self.control
                    else f"Detected {detected_prbs.size} PRBs above threshold (no control active)"
                ),
                control_fired=self.control,
            )
            dapp_logger.info(
                f"Annotation: label={ann['label']} | "
                f"detector={ann.get('detector')} | "
                f"all_detected_prbs={ann['all_detected_prbs']} | "
                f"noise_threshold={ann['noise_threshold']} dB | "
                f"power_db_max={ann['power_db_max']:.1f} dB | "
                f"noise_floor_per_prb={ann.get('noise_floor_per_prb')} | "
                f"snr_db_per_prb={ann.get('snr_db_per_prb')}"
            )
            if hasattr(self, '_ground_truth_label'):
                with self._ground_truth_lock:
                    gt = self._ground_truth_label if detected_prbs.size > 0 else "no_rfi"
                ann["ground_truth_label"] = gt
            with self._sample_idx_lock:
                if self.sample_idx is not None:
                    self.iq_saver.add_annotation(start_sample=self.sample_idx, **ann)

        # Dashboard shows all detected PRBs; zone overlays (when control is active) visually
        # distinguish the protected region from the region sent in the control message.
        if self.dashboard:
            self.demo_queue.put(("prb_list", detected_prbs))

    @override
    def _control_loop(self):
        # If no GUIs are enabled, just sleep to avoid busy-waiting
        if not (self.energyGui or self.iqPlotterGui or self.dashboard):
            time.sleep(1)
            return

        if self.energyGui:
            try:
                display_data = self.sig_queue.get(timeout=0.1)
                self.energyPlotter.process_iq_data(display_data)
            except queue.Empty:
                pass
            except Exception:
                dapp_logger.exception("[SPECTRUM] Error in energyGui control loop")

        if self.iqPlotterGui:
            try:
                iq_data = self.iq_queue.get(timeout=0.1)
                self.iqPlotter.process_iq_data(iq_data)
            except queue.Empty:
                pass
            except Exception:
                dapp_logger.exception("[SPECTRUM] Error in iqPlotterGui control loop")

        if self.dashboard:
            try:
                message = self.demo_queue.get(timeout=0.1)
                self.demo.process_iq_data(message)
            except queue.Empty:
                pass
            except Exception:
                dapp_logger.exception("[SPECTRUM] Error in dashboard control loop")

    @override
    def _stop(self):
        if self.save_iqs:
            self.iq_saver.close()
        if self.dashboard:
            self.demo.stop()
