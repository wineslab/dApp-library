#!/usr/bin/env python3
"""
dApp for Spectrum Sharing
"""

__author__ = "Andrea Lacava"

import multiprocessing
import time
import numpy as np
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
                center_freq: float = 3.6192e9, num_prbs: int = 106, num_subcarrier_spacing: int = 30, e_sampling: bool = False, **kwargs):
        super().__init__(id=id, link=link, transport=transport, **kwargs) 

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
        self.e3_interface.add_callback(self.get_iqs_from_ran)
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

    def get_iqs_from_ran(self, data):
        dapp_logger.debug(f'Triggered callback')
        if self.save_iqs:
            dapp_logger.debug("I will write on the logfile iqs")
            self.save_counter += 1
            self.iq_save_file.write(data)
            self.iq_save_file.flush()
            if self.save_counter > self.limit_per_file:
                self.iq_save_file.close()
                self.iq_save_file = open(f"{LOG_DIR}/iqs_{int(time.time())}.bin", "ab")
               
        iq_arr = np.frombuffer(data, dtype=np.int16)
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
                prb_new = prb_blk_list.view(prb_blk_list.dtype.newbyteorder('>'))
                
                # Create the payload
                size = prb_blk_list.size.to_bytes(2,'little')
                prbs_to_send = prb_new.tobytes(order="C")
                
                # Schedule the delivery
                self.e3_interface.schedule_control(size+prbs_to_send)

                if self.energyGui:
                    self.sig_queue.put(abs_iq_av_db)
                
                if self.dashboard:
                    self.demo_queue.put(("prb_list", prb_blk_list))

                # Reset the variables
                self.abs_iq_av = np.zeros(self.fft_size)
                self.control_count = 1  

    def _control_loop(self):
        if self.energyGui:
            abs_iq_av_db = self.sig_queue.get()
            self.energyPlotter.process_iq_data(abs_iq_av_db)

        if self.iqPlotterGui:
            iq_data = self.iq_queue.get()
            self.iqPlotter.process_iq_data(iq_data)

        if self.dashboard:
            message = self.demo_queue.get()
            self.demo.process_iq_data(message)

    def _stop(self):        
        if self.save_iqs:
            self.iq_save_file.close()
        
        if self.dashboard:
            self.demo.stop()
