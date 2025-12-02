#!/usr/bin/env python3
"""
dApp for Spectrum Sharing
"""

__author__ = "Andrea Lacava"

import multiprocessing
import queue
import time
import os
import numpy as np
import asn1tools
# np.set_printoptions(threshold=sys.maxsize)

from dapp.dapp import DApp
from e3interface.e3_logging import dapp_logger, LOG_DIR

class SpectrumSharingDApp(DApp):

    ###  Default Configuration ###
    # gNB runs with BW = ~40 MHz
    # Center frequency = 3.6192 GHz
    # OAI sampling frequency 46.08e6 # not useful for visualization
    # No SRS for now, hence in gNB config file set do_SRS = 0
    # gNB->frame_parms.ofdm_symbol_size = 1536 # FFT size (even though is called ofdm_symbol_size) for USRP, with -E flag (3/4 sampling)
    # gNB->frame_parms.ofdm_symbol_size = 2048 # FFT size (even though is called ofdm_symbol_size) for RUs, with no -E flag
    # Noise floor threshold needs to be calibrated according to the RU
    # We receive the symbols, average them over 63 frames and do PRB thresholding.

    def __init__(self, id: int = 1, link: str = 'posix', transport: str = 'ipc', noise_floor_threshold: int = 53, save_iqs: bool = False, control: bool = False,
                center_freq: float = 3.6192e9, num_prbs: int = 106, num_subcarrier_spacing: int = 30, e_sampling: bool = False, 
                encoding_method: str = "asn1", sampling_threshold: int = 5, **kwargs):
        super().__init__(id=id, link=link, transport=transport, encoding_method=encoding_method, **kwargs) 

        # RAN FUNCTION IDS OF INTEREST FOR THIS DAPP is 1 (IQs + PRB Control)
        # This might change soon and PRB control might become 2
        self.RAN_FUNCTION_ID = 1

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
        self.e3_interface.add_callback(self.dapp_id, self.get_iqs_from_ran)
        self.sampling_threshold = sampling_threshold
        if self.save_iqs:
            self.iq_save_file = open(f"{LOG_DIR}/iqs_{int(time.time())}.bin", "ab")
            self.save_counter = 0
            self.limit_per_file = 200
        self.control = control
        dapp_logger.info(f"Control is {'not ' if not self.control else ''}active")

        self.energyGui = kwargs.get('energyGui', False)
        self.iqPlotterGui = kwargs.get('iqPlotterGui', False)
        self.dashboard = kwargs.get('dashboard', False)

        self.control_count = 1
        self.abs_iq_av = np.zeros(self.fft_size)

        if self.energyGui:
            from visualization.energy import EnergyPlotter
            self.sig_queue = multiprocessing.Queue() 
            self.energyPlotter = EnergyPlotter(self.fft_size, bw=self.bw, center_freq=self.center_freq) 

        if self.iqPlotterGui:
            from visualization.iq import IQPlotter
            self.iq_queue = multiprocessing.Queue() 
            self.iqPlotter = IQPlotter(buffer_size=500, fft_size=self.fft_size, bw=self.bw, center_freq=self.center_freq)    

        if self.dashboard:
            from visualization.dashboard import Dashboard
            self.demo_queue = multiprocessing.Queue()
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
                # Future: Initialize JSON encoder
                self.spectrum_encoder = None
                dapp_logger.error("JSON encoding not yet implemented")
                raise NotImplementedError("JSON encoding not yet implemented")
            case _:
                raise ValueError(f"Unsupported encoding method: {self.encoding_method}")

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

    def create_prb_blacklist_control(self, blacklisted_prbs: bytes, prb_count: int,
                                     update_sampling: bool = False,
                                     validity_period: int = None) -> bytes:
        """Create a PRB blacklist control message
        
        Args:
            blacklisted_prbs: Raw PRB data as bytes (properly ordered)
            prb_count: Size of the list
            update_sampling: Check if the sampling threshold if the I/Qs should be updated (optional)
            validity_period: How long this blacklist is valid in seconds (optional)
            
        Returns:
            Encoded bytes for E3-ControlAction.actionData
        """
        control_data = {
            "blacklistedPRBs": blacklisted_prbs,
            "prbCount": prb_count
        }

        if update_sampling:
            control_data["samplingThreshold"] = self.sampling_threshold
                
        if validity_period is not None:
            control_data["validityPeriod"] = validity_period
        
        dapp_logger.debug(control_data)
        
        return self._encode_spectrum_message("Spectrum-PRBBlacklistControl", control_data)

    def create_prb_blacklist_report(self, blacklisted_prbs: bytes, prb_count: int) -> bytes:
        """Create a PRB blacklist report message

        Args:
            blacklisted_prbs: Raw PRB data as bytes (properly ordered)
            prb_count: Size of the list
            
        Returns:
            Encoded bytes for E3-DAppReport.reportData
        """
        report_data = {
            "blacklistedPRBs": blacklisted_prbs,
            "prbCount": prb_count
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
            # Future: Implement JSON encoding
            import json
            return json.dumps(data).encode('utf-8')
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
            data: Encoded bytes from E3-ControlAction.actionData
            
        Returns:
            Dictionary containing the decoded control data
        """
        return self._decode_spectrum_message("Spectrum-ConfigControl", data)

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
            # Future: Implement JSON decoding
            raise NotImplementedError("Json not implemented yet")
            import json
            return json.loads(data.decode('utf-8'))
        else:
            raise ValueError(f"Unsupported encoding method: {self.encoding_method}")

    def get_iqs_from_ran(self, dapp_identifier, data):
        dapp_logger.debug(f'Triggered callback for dApp {dapp_identifier}')
        indication_message = self.decode_iq_data_indication(data)
        iqs_raw = indication_message["iqSamples"]
        sample_count = indication_message["sampleCount"]
        timestamp = indication_message.get("timestamp", None)
        dapp_logger.debug(f"Received {sample_count} samples with timestamp {timestamp}")
        
        if self.save_iqs:
            dapp_logger.debug("I will write on the logfile iqs")
            self.save_counter += 1
            self.iq_save_file.write(iqs_raw)
            self.iq_save_file.flush()
            if self.save_counter > self.limit_per_file:
                self.iq_save_file.close()
                self.iq_save_file = open(f"{LOG_DIR}/iqs_{int(time.time())}.bin", "ab")
               
        iq_arr = np.frombuffer(iqs_raw, dtype=np.int16)
        dapp_logger.debug(f"Shape of iq_arr {iq_arr.shape}")

        if self.iqPlotterGui:
            self.iq_queue.put(iq_arr)

        if self.dashboard:
            self.demo_queue.put(("iq_data", iq_arr))

        if self.control:
            dapp_logger.debug("Start control operations")
            iq_comp = iq_arr[::2] + iq_arr[1::2] * 1j
            dapp_logger.debug(f"Shape of iq_comp {iq_comp.shape}")
            abs_iq = np.abs(iq_comp).astype(float)
            dapp_logger.debug(f"After iq division self.abs_iq_av: {self.abs_iq_av.shape} abs_iq: {abs_iq.shape}")
            self.abs_iq_av += abs_iq
            self.control_count += 1
            dapp_logger.debug(f"Control count is: {self.control_count}")

            if self.control_count == self.average_over_frames:
                abs_iq_av_db =  20 * np.log10(1 + (self.abs_iq_av/(self.average_over_frames)))
                abs_iq_av_db_offset_correct = np.append(abs_iq_av_db[self.first_carrier_offset:self.fft_size],abs_iq_av_db[0:self.first_carrier_offset])
                dapp_logger.info(f'--- AVG VALUES ----')
                dapp_logger.info(f'abs_iq_av_db_offset_correct: {abs_iq_av_db_offset_correct.mean()}')
                dapp_logger.info(f'--- MAX VALUES ----')
                dapp_logger.info(f'abs_iq_av_db_offset_correct: {abs_iq_av_db_offset_correct.max()}')
                # last_5_max_abs_iq_av_db_offset_correct = np.sort(abs_iq_av_db_offset_correct)[-20:]
                # dapp_logger.info(f'Last 20 max values (abs_iq_av_db_offset_correct): {last_5_max_abs_iq_av_db_offset_correct}')

                # PRB blocking based on the noise floor threshold
                f_ind = np.arange(self.fft_size)
                blklist_sub_carrier = f_ind[abs_iq_av_db_offset_correct > self.noise_floor_threshold]
                np.sort(blklist_sub_carrier)
                dapp_logger.info(f'blklist_sub_carrier: {blklist_sub_carrier}')
                prb_blk_list = np.unique((np.floor(blklist_sub_carrier/self.num_consecutive_subcarriers_for_prb))).astype(np.uint16)
                dapp_logger.info(f'prb_blk_list: {prb_blk_list}')
                prb_blk_list = prb_blk_list[prb_blk_list > self.prb_thrs]
                # prb_blk_list = np.array([76, 77,  78,  79, 80, 81, 82, 83], dtype=np.uint16) # 76, 77,  78,  79, 80, 81, 82, 83
                # prb_blk_list = np.arange(start=76, stop=140, dtype=np.uint16)
                dapp_logger.info(f"Blacklisted prbs ({prb_blk_list.size}): {prb_blk_list}")
                
                # Apply custom control logic if callback is set
                update_sampling = False
                if self._sampling_threshold_control_callback is not None:
                    try:
                        update_sampling, new_sampling_threshold = self._sampling_threshold_control_callback(
                            prb_blk_list, abs_iq_av_db_offset_correct
                        )
                        if update_sampling:
                            self.sampling_threshold = new_sampling_threshold
                            dapp_logger.info(f"Custom logic updated sampling threshold to {self.sampling_threshold}")
                    except Exception:
                        dapp_logger.exception(f"Error in custom control callback")
                        update_sampling = False
                
                prb_new = prb_blk_list.view(prb_blk_list.dtype.newbyteorder('>'))
                prbs_to_send = prb_new.tobytes(order="C")

               
                control_payload = self.create_prb_blacklist_control(blacklisted_prbs=prbs_to_send,
                                                                    prb_count=prb_blk_list.size,
                                                                    update_sampling=update_sampling)
                report_payload = self.create_prb_blacklist_report(
                    blacklisted_prbs=prbs_to_send,
                    prb_count=prb_blk_list.size
                )
                
                self.e3_interface.schedule_control(dappId=self.dapp_id, ranFunctionId=self.RAN_FUNCTION_ID, actionData=control_payload)
                self.e3_interface.schedule_report(dappId=self.dapp_id, ranFunctionId=self.RAN_FUNCTION_ID, reportData=report_payload)

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

        try:
            if self.energyGui:    
                abs_iq_av_db = self.sig_queue.get(timeout=0.1)
                self.energyPlotter.process_iq_data(abs_iq_av_db)

            if self.iqPlotterGui:
                iq_data = self.iq_queue.get(timeout=0.1)
                self.iqPlotter.process_iq_data(iq_data)
    
            if self.dashboard:
                message = self.demo_queue.get(timeout=0.1)
                self.demo.process_iq_data(message)
        except queue.Empty:
            pass # This is allowed
        except Exception:
            dapp_logger.exception("[SPECTRUM] Error in the control loop")

    def _stop(self):        
        if self.save_iqs:
            self.iq_save_file.close()
        
        if self.dashboard:
            self.demo.stop()
