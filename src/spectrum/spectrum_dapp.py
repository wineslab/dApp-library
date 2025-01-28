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

    ###  Configuration ###
    # gNB runs with BW = 40 MHz, with -E (3/4 sampling)
    # Center frequency = 3.6192 GHz
    # OAI sampling frequency 46.08e6
    # No SRS for now, hence in gNB config file set do_SRS = 0
    # gNB->frame_parms.ofdm_symbol_size = 1536
    # gNB->frame_parms.first_carrier_offset = 900
    # Noise floor threshold needs to be calibrated
    # We receive the symbols and average them over some frames, and do thresholding.

    def __init__(self, id: int = 1, ota: bool = False, save_iqs: bool = False, control: bool = False, link: str = 'posix', transport:str = 'uds', **kwargs):
        super().__init__(link=link, transport=transport, **kwargs) 

        self.bw = 40.08e6  # Bandwidth in Hz
        self.center_freq = 3.6192e9 # Center frequency in Hz
        self.First_carrier_offset = 900
        self.Num_car_prb = 12
        self.prb_thrs = 75 # This avoids blacklisting PRBs where the BWP is scheduled (it’s a workaround bc the UE and gNB would not be able to communicate anymore, a cleaner fix is to move the BWP if needed or things like that)
        self.FFT_SIZE = 1536  
        self.Average_over_frames = 63
        
        if ota:
            dapp_logger.info(f'Using OTA configuration')
            self.Noise_floor_threshold = 20 # this really depends on the RF conditions and should be carefully calibrated
        else: # Colosseum
            dapp_logger.info(f'Using Colosseum configuration')
            self.Noise_floor_threshold = 53

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
        self.demoGui = kwargs.get('demoGui', False)

        self.control_count = 1
        self.abs_iq_av = np.zeros(self.FFT_SIZE)

        if self.energyGui:
            from visualization.energy import EnergyPlotter
            self.sig_queue = multiprocessing.Queue() 
            self.energyPlotter = EnergyPlotter(self.FFT_SIZE, bw=self.bw, center_freq=self.center_freq) 

        if self.iqPlotterGui:
            from visualization.iq import IQPlotter
            self.iq_queue = multiprocessing.Queue() 
            iq_size = self.FFT_SIZE * 2 # double the size of ofdm_symbol_size since real and imaginary parts are interleaved
            self.iqPlotter = IQPlotter(buffer_size=500, iq_size=iq_size, bw=self.bw, center_freq=self.center_freq)    

        if self.demoGui:
            from visualization.dashboard import DemoGui
            self.demo_queue = multiprocessing.Queue()
            iq_size = self.FFT_SIZE * 2 # double the size of ofdm_symbol_size since real and imaginary parts are interleaved
            self.demo = DemoGui(buffer_size=100, iq_size=iq_size) 

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

        if self.demoGui:
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

            if self.iqPlotterGui:
                self.iq_queue.put(iq_arr)

            if self.control_count == self.Average_over_frames:
                abs_iq_av_db =  20 * np.log10(1 + (self.abs_iq_av/(self.Average_over_frames)))
                abs_iq_av_db_offset_correct = np.append(abs_iq_av_db[self.First_carrier_offset:self.FFT_SIZE],abs_iq_av_db[0:self.First_carrier_offset])
                dapp_logger.info(f'--- AVG VALUES ----')
                dapp_logger.info(f'abs_iq_av_db_offset_correct: {abs_iq_av_db_offset_correct.mean()}')
                dapp_logger.info(f'--- MAX VALUES ----')
                dapp_logger.info(f'abs_iq_av_db_offset_correct: {abs_iq_av_db_offset_correct.max()}')
                # last_5_max_abs_iq_av_db_offset_correct = np.sort(abs_iq_av_db_offset_correct)[-20:]
                # dapp_logger.info(f'Last 20 max values (abs_iq_av_db_offset_correct): {last_5_max_abs_iq_av_db_offset_correct}')


                # PRB blocking based on the noise floor threshold
                f_ind = np.arange(self.FFT_SIZE)
                blklist_sub_carrier = f_ind[abs_iq_av_db_offset_correct > self.Noise_floor_threshold]
                np.sort(blklist_sub_carrier)
                dapp_logger.info(f'blklist_sub_carrier: {blklist_sub_carrier}')
                prb_blk_list = np.unique((np.floor(blklist_sub_carrier/self.Num_car_prb))).astype(np.uint16)
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
                
                if self.demoGui:
                    self.demo_queue.put(("prb_list", prb_blk_list))

                # Reset the variables
                self.abs_iq_av = np.zeros(self.FFT_SIZE)
                self.control_count = 1  

    def _control_loop(self):
        if self.energyGui:
            abs_iq_av_db = self.sig_queue.get()
            self.energyPlotter.process_iq_data(abs_iq_av_db)
        if self.iqPlotterGui:
            iq_data = self.iq_queue.get()
            self.iqPlotter.process_iq_data(iq_data)
        if self.demoGui:
            message = self.demo_queue.get()
            self.demo.process_iq_data(message)

    def _stop(self):        
        if self.save_iqs:
            self.iq_save_file.close()
        
        if self.demoGui:
            self.demo.stop()
