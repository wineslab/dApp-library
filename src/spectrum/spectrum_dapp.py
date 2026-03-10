#!/usr/bin/env python3
"""
dApp for Spectrum Sharing
"""

__author__ = "Andrea Lacava"

import threading
import queue
import time
import os
import json
# from typing import override
import numpy as np
import asn1tools
import jsonschema
# np.set_printoptions(threshold=sys.maxsize)

from dapp.dapp import DApp
from e3interface.e3_encoder import JsonE3Encoder
from e3interface.e3_logging import dapp_logger, LOG_DIR

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

    ###  Default Configuration ###
    # gNB runs with BW = ~40 MHz
    # Center frequency = 3.6192 GHz
    # OAI sampling frequency 46.08e6 # not useful for visualization
    # No SRS for now, hence in gNB config file set do_SRS = 0
    # gNB->frame_parms.ofdm_symbol_size = 1536 # FFT size (even though is called ofdm_symbol_size) for USRP, with -E flag (3/4 sampling)
    # gNB->frame_parms.ofdm_symbol_size = 2048 # FFT size (even though is called ofdm_symbol_size) for RUs, with no -E flag
    # Noise floor threshold needs to be calibrated according to the RU
    # We receive the symbols, average them over 63 frames and do PRB thresholding.

    def __init__(self, dapp_name: str = DAPP_NAME, dapp_version: str = DAPP_VERSION,
                 vendor: str = VENDOR, e3ap_protocol_version: str = E3AP_PROTOCOL_VERSION,
                 link: str = 'posix', transport: str = 'ipc', 
                 noise_floor_threshold: int = 53, save_iqs: bool = False, control: bool = False,
                 center_freq: float = 3.6192e9, num_prbs: int = 106, num_subcarrier_spacing: int = 30, 
                 e_sampling: bool = False, encoding_method: str = "asn1", sampling_threshold: int = 5, **kwargs):
        super().__init__(dapp_name=dapp_name, dapp_version=dapp_version, vendor=vendor,
                         e3ap_protocol_version=e3ap_protocol_version, link=link, transport=transport, 
                         encoding_method=encoding_method, **kwargs) 

        # Initialize spectrum encoder based on encoding method
        self._init_spectrum_encoder()

        # Custom control logic callback
        self._sampling_threshold_control_callback = None 

        # gNB radio configuration used to compute exact values
        self.num_consecutive_subcarriers_for_prb: int = 12 # Always fixed by the LTE/NR standard to 12
        self.num_prbs = num_prbs
        self.num_subcarrier_spacing = num_subcarrier_spacing # subcarrier spacing in kHz (FR1 is 30)
        self.ofdm_symbol_size = num_prbs * self.num_consecutive_subcarriers_for_prb
        self.bw = self.num_prbs * self.num_consecutive_subcarriers_for_prb * self.num_subcarrier_spacing * 1e3 # Bandwidth in Hz
        self.center_freq = center_freq # Center frequency in Hz
        self.fft_size = 2 ** int(np.ceil(np.log2(self.ofdm_symbol_size))) # Next power of 2 to include guard bands
        if e_sampling:
            self.fft_size = int(self.fft_size * 3 / 4) # 3/4 of sampling for USRPs (-E flag) 
        self.first_carrier_offset = self.fft_size - (self.ofdm_symbol_size // 2)

        # dApp configuration
        # The first 76 PRB usually have channels that should not be nulled (but we should investigate more options to enable this)
        self.prb_thrs = 75 # This avoids blacklisting PRBs where the BWP is scheduled (it’s a workaround since the UE and gNB would not be able to communicate anymore, a cleaner fix is to move the BWP if needed or things like that)  
        self.average_over_frames = 63
        self.noise_floor_threshold = noise_floor_threshold
        self.save_iqs = save_iqs
        self.sampling_threshold = sampling_threshold
        if self.save_iqs:
            from iq_saver.iq_saver import IQSaver
            # Calculate effective sample rate based on sampling_threshold
            # Each capture is: 10ms * sampling_threshold
            sample_rate = 100 # Hz since sensing is done once every 10ms
            dapp_logger.info(f"Sensing sample rate: {sample_rate:.2f} Hz (Each sensing is done 10ms")

            self.iq_saver = IQSaver(
                base_path=LOG_DIR,
                center_freq=self.center_freq,
                bandwidth=self.bw,
                sample_rate=sample_rate,
                annotation_flush_interval=10,
                hw_info=f"FFT:{self.fft_size}, PRBs:{self.num_prbs}, E-sampling:{e_sampling}",
                description=f"5G NR Uplink capture from SpectrumSharing dApp - RAN Function {self.RAN_FUNCTION_ID}",
                fft_size=self.fft_size,
                dtype="ci16_le",
                num_prbs=self.num_prbs,
                subcarrier_spacing_khz=self.num_subcarrier_spacing,
                sampling_threshold=self.sampling_threshold
            )
        self.control = control
        dapp_logger.info(f"Control is {'not ' if not self.control else ''}active")

        self.energyGui = kwargs.get('energyGui', False)
        self.iqPlotterGui = kwargs.get('iqPlotterGui', False)
        self.dashboard = kwargs.get('dashboard', False)

        self.control_count = 1
        self.abs_iq_av = np.zeros(self.fft_size)

        # Thread-safe sample_idx
        self._sample_idx_lock = threading.Lock()
        self.sample_idx = None

        if self.energyGui:
            from visualization.energy import EnergyPlotter
            self.sig_queue = queue.Queue()
            self.energyPlotter = EnergyPlotter(self.fft_size, bw=self.bw, center_freq=self.center_freq)

        if self.iqPlotterGui:
            from visualization.iq import IQPlotter
            self.iq_queue = queue.Queue()
            self.iqPlotter = IQPlotter(buffer_size=500, fft_size=self.fft_size, bw=self.bw, center_freq=self.center_freq)

        if self.dashboard:
            from visualization.dashboard import Dashboard
            self.demo_queue = queue.Queue()
            classifier = kwargs.get('classifier', None)
            self.demo = Dashboard(buffer_size=100, ofdm_symbol_size=self.ofdm_symbol_size, first_carrier_offset=self.first_carrier_offset,
                                bw=self.bw, center_freq=self.center_freq, num_prbs=num_prbs, classifier=classifier)

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
        """Set a custom control logic callback
        
        This method should be armed before starting the setup exchange
        The callback function will be invoked during the control loop and should have the signature:
            callback(prb_blacklist: np.ndarray, spectrum_data: np.ndarray) -> tuple[bool, int]
        
        Args:
            callback: A callable that takes:
                - prb_blacklist (np.ndarray): Array of blacklisted PRB indices
                - spectrum_data (np.ndarray): Averaged spectrum data in dB (offset corrected)
                
                Returns:
                - update_sampling (bool): Whether to update the sampling threshold
                - sampling_threshold (int): The new sampling threshold value (0-100)
        
        Example:
            def my_custom_logic(prb_blacklist, spectrum_data):
                # Custom logic here
                update = len(prb_blacklist) > 10
                new_threshold = 10 if update else 5
                return update, new_threshold
            
            dapp.set_sampling_threshold_control_logic(my_custom_logic)
        """
        if callback is not None and not callable(callback):
            raise ValueError("Callback must be callable")
        self._sampling_threshold_control_callback = callback
        dapp_logger.info(f"Custom control logic callback {'set' if callback else 'removed'}")

    def create_prb_blacklist_control(self, blacklisted_prbs: list[int],
                                     update_sampling: bool = False,
                                     validity_period: int = None) -> bytes:
        """Create a PRB blacklist control message
        
        Args:
            blacklisted_prbs: List of PRB indices
            update_sampling: Check if the sampling threshold if the I/Qs should be updated (optional)
            validity_period: How long this blacklist is valid in seconds (optional)
            
        Returns:
            Encoded bytes for E3-DAppControlAction.actionData
        """
        control_data = {
            "blacklistedPRBs": blacklisted_prbs
        }

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
        report_data = {
            "blacklistedPRBs": blacklisted_prbs
        }

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
                message_type,
                data.copy(),
                self._SPECTRUM_JSON_BINARY_FIELDS
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
            Dictionary containing the decoded indication data
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

    # @override
    def _decode_ran_function_data(self, data_bytes: bytes) -> dict | None:
        """Decode the `ranFunctionData` attached to a SetupResponse entry.

        Returns the decoded dict or None on failure.
        """
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
                message_type,
                decoded_data,
                self._SPECTRUM_JSON_BINARY_FIELDS
            )
        else:
            raise ValueError(f"Unsupported encoding method: {self.encoding_method}")

    # @override
    def _handle_xapp_control(self, dapp_identifier: int, data: bytes):       
        dapp_logger.info(f'Triggered control callback for dApp {dapp_identifier}')

        decoded = self._decode_spectrum_message("Spectrum-PRBBlockedControl", data)
        prb_blk_list = decoded["blockedPRBs"]
        dapp_logger.info(f"xApp control: blockedPRBs={prb_blk_list}")

        control_payload = self.create_prb_blacklist_control(blacklisted_prbs=prb_blk_list)
        self.e3_interface.schedule_control(dappId=self.dapp_id, ranFunctionId=self.RAN_FUNCTION_ID, controlId=self.CONTROL_ID[0], actionData=control_payload)
        dapp_logger.info(f"Sending Control to RAN: blockedPRBs={prb_blk_list}")
        # Add annotation to IQ recording
        if self.save_iqs and self.sample_idx is not None:
            dapp_logger.info(f"Add annotation")
            self.iq_saver.add_annotation(
                        start_sample=self.sample_idx,
                        label="prb_control",
                        comment=f"Blacklisted {len(prb_blk_list)} PRBs upon message from xApp",
                        timestamp=time.time(), # Since this is a control coming from the xApp it makes sense to save when it has been elapsed
                        prb_blacklist=prb_blk_list,
                        noise_threshold=self.noise_floor_threshold,
                        control_action="blacklist",
                    )
            dapp_logger.info(f"Annotation added")

    # @override
    def _handle_indication(self, dapp_identifier: int, data: bytes):
        indication_message = self.decode_iq_data_indication(data)
        iqs_raw = indication_message["iqSamples"]
        sample_count = indication_message["sampleCount"]
        timestamp = indication_message.get("timestamp", None)
        dapp_logger.debug(f"Received {sample_count} samples with timestamp {timestamp}")

        # Convert raw bytes to numpy array
        iq_arr = np.frombuffer(iqs_raw, dtype=np.int16)
        dapp_logger.debug(f"Shape of iq_arr {iq_arr.shape}")

        if self.iqPlotterGui:
            self.iq_queue.put(iq_arr)

        if self.dashboard:
            self.demo_queue.put(("iq_data", iq_arr))

        dapp_logger.debug("Start control operations")
        iq_comp = iq_arr[::2] + iq_arr[1::2] * 1j
        dapp_logger.debug(f"Shape of iq_comp {iq_comp.shape}")

        # Save IQ samples using SigMF-compliant IQSaver
        with self._sample_idx_lock:
            self.sample_idx = None
            if self.save_iqs:
                self.sample_idx = self.iq_saver.save_samples(iq_arr, timestamp=timestamp)

        abs_iq = np.abs(iq_comp).astype(float)
        dapp_logger.debug(f"After iq division self.abs_iq_av: {self.abs_iq_av.shape} abs_iq: {abs_iq.shape}")
        self.abs_iq_av += abs_iq
        self.control_count += 1
        dapp_logger.debug(f"Control count is: {self.control_count}")

        if self.control_count >= self.average_over_frames:
            abs_iq_av_db = 20 * np.log10(
                1 + (self.abs_iq_av / (self.average_over_frames))
            )
            abs_iq_av_db_offset_correct = np.append(
                abs_iq_av_db[self.first_carrier_offset : self.fft_size],
                abs_iq_av_db[0 : self.first_carrier_offset],
            )
            dapp_logger.info(f"--- AVG VALUES ----")
            dapp_logger.info(
                f"abs_iq_av_db_offset_correct: {abs_iq_av_db_offset_correct.mean()}"
            )
            dapp_logger.info(f"--- MAX VALUES ----")
            dapp_logger.info(
                f"abs_iq_av_db_offset_correct: {abs_iq_av_db_offset_correct.max()}"
            )
            # last_5_max_abs_iq_av_db_offset_correct = np.sort(abs_iq_av_db_offset_correct)[-20:]
            # dapp_logger.info(f'Last 20 max values (abs_iq_av_db_offset_correct): {last_5_max_abs_iq_av_db_offset_correct}')

            # PRB blocking based on the noise floor threshold
            f_ind = np.arange(self.fft_size)
            blklist_sub_carrier = f_ind[
                abs_iq_av_db_offset_correct > self.noise_floor_threshold
            ]
            dapp_logger.info(f"blklist_sub_carrier: {blklist_sub_carrier}")
            prb_blk_list = np.unique(
                (
                    np.floor(
                        blklist_sub_carrier / self.num_consecutive_subcarriers_for_prb
                    )
                )
            ).astype(np.uint16)
            dapp_logger.info(f"prb_blk_list: {prb_blk_list}")
            prb_blk_list = prb_blk_list[prb_blk_list > self.prb_thrs]
            # prb_blk_list = np.array([76, 77,  78,  79, 80, 81, 82, 83], dtype=np.uint16) # 76, 77,  78,  79, 80, 81, 82, 83
            # prb_blk_list = np.arange(start=76, stop=140, dtype=np.uint16)
            dapp_logger.info(f"Blacklisted prbs ({prb_blk_list.size}): {prb_blk_list}")

            # Apply custom control logic if callback is set
            update_sampling = False
            if self._sampling_threshold_control_callback is not None:
                try:
                    update_sampling, new_sampling_threshold = (
                        self._sampling_threshold_control_callback(
                            prb_blk_list, abs_iq_av_db_offset_correct
                        )
                    )
                    if update_sampling:
                        self.sampling_threshold = new_sampling_threshold
                        dapp_logger.info(
                            f"Custom logic updated sampling threshold to {self.sampling_threshold}"
                        )
                        # Update IQSaver sample rate to reflect new sampling threshold
                        if self.save_iqs:
                            new_sample_rate = 1 / (0.01 * self.sampling_threshold)
                            self.iq_saver.update_sample_rate(
                                new_sample_rate,
                                sampling_threshold=self.sampling_threshold,
                            )
                            dapp_logger.info(
                                f"Updated IQ saver sample rate to {new_sample_rate:.2f} Hz (sampling_threshold={self.sampling_threshold})"
                            )
                except Exception:
                    dapp_logger.exception(f"Error in custom control callback")
                    update_sampling = False

            prb_list_for_asn = prb_blk_list.astype(int).tolist()

            report_payload = self.create_prb_blacklist_report(
                blacklisted_prbs=prb_list_for_asn
            )

            self.e3_interface.schedule_report(
                dappId=self.dapp_id,
                ranFunctionId=self.RAN_FUNCTION_ID,
                reportData=report_payload,
            )

            if self.control:
                control_payload = self.create_prb_blacklist_control(
                    blacklisted_prbs=prb_list_for_asn, update_sampling=update_sampling
                )
                self.e3_interface.schedule_control(
                    dappId=self.dapp_id,
                    ranFunctionId=self.RAN_FUNCTION_ID,
                    controlId=self.CONTROL_ID[0],
                    actionData=control_payload,
                )

                # Add annotation to IQ recording
                with self._sample_idx_lock:
                    if self.save_iqs and self.sample_idx is not None:
                        dapp_logger.info(f"Add annotation")
                        self.iq_saver.add_annotation(
                                    start_sample=self.sample_idx,
                                    label="prb_control",
                                    timestamp=timestamp,
                                    comment=f"Blacklisted {prb_blk_list.size} PRBs due to interference",
                                    prb_blacklist=prb_blk_list.tolist(),
                                    noise_threshold=self.noise_floor_threshold,
                                    control_action="blacklist",
                                )
                        dapp_logger.info(f"Annotation added")

            if self.energyGui:
                self.sig_queue.put(abs_iq_av_db)

            if self.dashboard:
                self.demo_queue.put(("prb_list", prb_blk_list))

            # Reset the variables
            self.abs_iq_av = np.zeros(self.fft_size)
            self.control_count = 1

    def _control_loop(self):
        # If no GUIs are enabled, just sleep to avoid busy-waiting
        if not (self.energyGui or self.iqPlotterGui or self.dashboard):
            time.sleep(1)
            return

        if self.energyGui:
            try:
                abs_iq_av_db = self.sig_queue.get(timeout=0.1)
                self.energyPlotter.process_iq_data(abs_iq_av_db)
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

    def _stop(self):        
        if self.save_iqs:
            self.iq_saver.close()

        if self.dashboard:
            self.demo.stop()
