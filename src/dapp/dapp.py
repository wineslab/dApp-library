#!/usr/bin/env python3
"""
Example of a dApp
"""

__author__ = "Andrea Lacava"
__version__ = "0.1.0"
__license__ = "MIT"

from abc import ABC
import argparse
import queue
import socket
import threading
import time
import sys

import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from plotter import IQPlotter, EnergyPlotter
from e3_interface import E3Interface
import logging
import matplotlib
matplotlib.use('TkAgg')
import os

LOG_DIR = ('.' if os.geteuid() != 0 else '') + '/logs/'

# Configure logging for DApp
dapp_logger = logging.getLogger("dapp_logger")
dapp_logger.setLevel(logging.INFO)
dapp_handler = logging.FileHandler(f"{LOG_DIR}/dapp.log")
dapp_handler.setLevel(logging.INFO)
dapp_formatter = logging.Formatter("[dApp] [%(created)f] %(levelname)s - %(message)s")
dapp_handler.setFormatter(dapp_formatter)
dapp_logger.addHandler(dapp_handler)


class DApp(ABC):

    e3_interface: E3Interface
    control: bool
    counter: int
    limit_per_file: int
    DAPP_SOCKET_PATH = "/tmp/dapps/dapp_socket"

    ###  Configuration ###
    # gNB runs with BW = 40 MHz, with -E (3/4 sampling)
    # OAI sampling frequency 46.08e6 
    # No SRS for now, hence in gNB config file set do_SRS = 0
    # gNB->frame_parms.ofdm_symbol_size = 1536 # ON COLOSSEUM
    # gNB->frame_parms.ofdm_symbol_size = 2048 # ON OTA
    # gNB->frame_parms.first_carrier_offset = 900
    # Noise floor threshold needs to be calibrated
    # We receive the symbols and average them over some frames, and do thresholding.

    def __init__(self, ota: bool = False, control: bool = False, **kwargs):
        super().__init__()
        self.e3_interface = E3Interface()
        self.stop_event = threading.Event()
        if ota:
            dapp_logger.info(f'Using OTA configuration')
            self.bw = 40.08e6  # Bandwidth in Hz
            self.center_freq = 3.288e9 # Center frequency in Hz
            self.FFT_SIZE = 2048
            self.Noise_floor_threshold = 15
            self.First_carrier_offset = 0
            self.Average_over_frames = 127
            self.Num_car_prb = 106
            self.prb_thrs = 0 # This avoids blacklisting PRBs where the BWP is scheduled (it’s a workaround bc the UE and gNB would not be able to communicate anymore, a cleaner fix is to move the BWP if needed or things like that)
        else: # Colosseum
            dapp_logger.info(f'Using Colosseum configuration')
            self.bw = 40.08e6 # Bandwidth in Hz
            self.center_freq = 3.6e9 # Center frequency in Hz
            self.FFT_SIZE = 1536
            self.Noise_floor_threshold = 48
            self.First_carrier_offset = 900
            self.Average_over_frames = 127
            self.Num_car_prb = 12
            self.prb_thrs = 75 # See above for explanation

        self.iq_save_file = open(f"{LOG_DIR}/iqs_{int(time.time())}.bin", "ab")
        # Initialize the singleton instance
        self.e3_interface.add_callback(self.save_iq_samples)
        self.counter = 0
        self.limit_per_file = 200
        self.control = control
        dapp_logger.info(f"Control is {'not ' if not self.control else ''}active")

        self.energyGui = kwargs.get('energyGui', False)
        self.iqPlotterGui = kwargs.get('iqPlotterGui', False)

        self.control_count = 1
        self.abs_iq_av = np.zeros(self.FFT_SIZE)
        
        if self.energyGui:
            self.sig_queue = queue.Queue() 
            self.energyPlotter = EnergyPlotter(self.FFT_SIZE)
        if self.iqPlotterGui:
            self.iq_queue = queue.Queue() 
            iq_size = self.FFT_SIZE * 2 # double the size of ofdm_symbol_size since real and imaginary parts are interleaved
            self.iqPlotter = IQPlotter(buffer_size=100, iq_size=iq_size, bw=self.bw, center_freq=self.center_freq)    

        if self.control:            
            # Creating a client to send PRB updates and apply control
            self.prb_updates_socket = None

            while self.prb_updates_socket is None:
                try:
                    self.prb_updates_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    self.prb_updates_socket.connect(self.DAPP_SOCKET_PATH)
                except (FileNotFoundError, ConnectionRefusedError):
                    self.prb_updates_socket = None
                    dapp_logger.info("gNB server for control is not up yet, sleeping for 5 seconds")
                    time.sleep(5)

    def save_iq_samples(self, data):
        dapp_logger.debug("I will write on the logfile iqs")
        self.counter += 1
        self.iq_save_file.write(data)
        self.iq_save_file.flush()
        if self.counter > self.limit_per_file:
            self.iq_save_file.close()
            self.iq_save_file = open(f"{LOG_DIR}/iqs_{int(time.time())}.bin", "ab")
        
        if self.control:
            dapp_logger.debug("Start control operations")
            iq_arr = np.frombuffer(data, np.int16)
            iq_comp = iq_arr[::2] + iq_arr[1::2] * 1j
            abs_iq = abs(iq_comp).astype(float)
            dapp_logger.debug(f"After iq division self.abs_iq_av: {self.abs_iq_av.shape} abs_iq: {abs_iq.shape}")
            self.abs_iq_av += abs_iq
            self.control_count += 1
            dapp_logger.debug(f"Control count is: {self.control_count}")

            if self.control_count == self.Average_over_frames:
                abs_iq_av_db =  20 * np.log10(1 + (self.abs_iq_av/(self.Average_over_frames)))
                abs_iq_av_db_offset_correct = np.append(abs_iq_av_db[self.First_carrier_offset:self.FFT_SIZE],abs_iq_av_db[0:self.First_carrier_offset])
                dapp_logger.info(f'--- AVG VALUES ----')
                dapp_logger.info(f'abs_iq_av_db: {abs_iq_av_db.mean()}')
                dapp_logger.info(f'abs_iq_av_db_offset_correct: {abs_iq_av_db_offset_correct.mean()}')
                dapp_logger.info(f'--- MAX VALUES ----')
                dapp_logger.info(f'abs_iq_av_db: {abs_iq_av_db.max()}')
                dapp_logger.info(f'abs_iq_av_db_offset_correct: {abs_iq_av_db_offset_correct.max()}')
                
                # Blacklisting based on the noise floor threshold
                f_ind = np.arange(self.FFT_SIZE)
                blklist_sub_carier = f_ind[abs_iq_av_db_offset_correct > self.Noise_floor_threshold]
                np.sort(blklist_sub_carier)
                prb_blk_list = np.unique((np.floor(blklist_sub_carier/self.Num_car_prb))).astype(np.uint16)
                prb_blk_list = prb_blk_list[prb_blk_list > self.prb_thrs]
                dapp_logger.info(f"Blacklisted prbs: {prb_blk_list}")
                prb_new = prb_blk_list.view(prb_blk_list.dtype.newbyteorder('>'))
                t1 = threading.Thread(target=self.prb_update, args=(prb_new, prb_blk_list.size,))
                t1.start() 

                if self.energyGui:
                    self.sig_queue.put(abs_iq_av_db)
                if self.iqPlotterGui:
                    self.iq_queue.put(iq_arr)

                # reset the variables
                self.abs_iq_av = np.zeros(self.FFT_SIZE)
                self.control_count = 1  

    def prb_update(self, prb_blk_list: np.array, n):
        if not self.control:
            return
        array2 = prb_blk_list.tobytes(order="C")
        array1 = n.to_bytes(2, "little")
        self.prb_updates_socket.send(array1 + array2)

    def control_loop(self):
        dapp_logger.debug(f"Start control loop")
        while not self.stop_event.is_set():
            try:
                if self.energyGui:
                    abs_iq_av_db = self.sig_queue.get()
                    self.energyPlotter.process_iq_data(abs_iq_av_db)
                if self.iqPlotterGui:                  
                    iq_data = self.iq_queue.get()
                    self.iqPlotter.process_iq_data(iq_data)
            except KeyboardInterrupt:
                dapp_logger.debug("Keyboard interrupt")
                self.stop_event.set()

    def __del__(self):
        self.stop_event.set()
        if self.control:
            # close connection socket with the client
            self.prb_updates_socket.close()
            dapp_logger.info("Connection to client for control closed")

        self.e3_interface.stop_server()
        dapp_logger.info("Stopped server")
        self.iq_save_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dApp example")
    parser.add_argument('--ota', action='store_true', default=False, help="Specify if this is OTA or on Colosseum")
    parser.add_argument('--control', action='store_true', default=False, help="Set whether to perform control of PRB")
    parser.add_argument('--energy-gui', action='store_true', default=False, help="Set whether to enable the energy GUI")
    parser.add_argument('--iq-plotter-gui', action='store_true', default=False, help="Set whether to enable the IQ Plotter GUI")
    args = parser.parse_args()
    
    dapp = DApp(ota=args.ota, control=args.control, energyGui=args.energy_gui, iqPlotterGui=args.iq_plotter_gui)
    dapp.control_loop()