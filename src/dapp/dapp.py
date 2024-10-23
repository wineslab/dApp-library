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
from e3_interface import E3Interface
import logging
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
    DAPP_UDS_SOCKET_PATH = "/tmp/dapps/dapp_socket"

    ###  Configuration ###
    # gNB runs with BW = 40 MHz, with -E (3/4 sampling)
    # Center frequency = 3.6192 GHz
    # OAI sampling frequency 46.08e6
    # No SRS for now, hence in gNB config file set do_SRS = 0
    # gNB->frame_parms.ofdm_symbol_size = 1536
    # gNB->frame_parms.first_carrier_offset = 900
    # Noise floor threshold needs to be calibrated
    # We receive the symbols and average them over some frames, and do thresholding.

    def __init__(self, ota: bool = False, save_iqs: bool = False, control: bool = False, **kwargs):
        super().__init__()
        self.profile = kwargs.get('profile', False)
        self.e3_interface = E3Interface(ota=ota, profile=self.profile)
        self.stop_event = threading.Event()

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
            from energy_plotter import EnergyPlotter
            self.sig_queue = queue.Queue() 
            self.energyPlotter = EnergyPlotter(self.FFT_SIZE, bw=self.bw, center_freq=self.center_freq) 

        if self.iqPlotterGui:
            from iq_plotter import IQPlotter
            self.iq_queue = queue.Queue() 
            iq_size = self.FFT_SIZE * 2 # double the size of ofdm_symbol_size since real and imaginary parts are interleaved
            self.iqPlotter = IQPlotter(buffer_size=500, iq_size=iq_size, bw=self.bw, center_freq=self.center_freq)    

        if self.demoGui:
            from demo_plotter import DemoGui
            self.demo_queue = queue.Queue()
            iq_size = self.FFT_SIZE * 2 # double the size of ofdm_symbol_size since real and imaginary parts are interleaved
            self.demo = DemoGui(buffer_size=100, iq_size=iq_size) 

        if self.control:            
            # Creating a client to send PRB updates and apply control
            self.prb_updates_socket = None
            socket_type = socket.AF_UNIX if ota else socket.AF_INET
            connection_target = self.DAPP_UDS_SOCKET_PATH if ota else ("127.0.0.1", 9999)
            dapp_logger.info(f"{'Control socket is using AF_UNIX' if socket_type == 1 else 'AF_INET'} {connection_target}")
            
            while self.prb_updates_socket is None:
                try:
                    self.prb_updates_socket = socket.socket(socket_type, socket.SOCK_STREAM)
                    self.prb_updates_socket.connect(connection_target)
                except (FileNotFoundError, ConnectionRefusedError):
                    self.prb_updates_socket = None
                    dapp_logger.info("gNB server for control is not up yet, sleeping for 5 seconds")
                    time.sleep(5)

            self.prb_queue = queue.Queue() 
            self.control_thread = threading.Thread(target=self.prb_update_task)
            self.control_thread.start() 

    def get_iqs_from_ran(self, data):
        if self.save_iqs:
            dapp_logger.debug("I will write on the logfile iqs")
            self.save_counter += 1
            self.iq_save_file.write(data)
            self.iq_save_file.flush()
            if self.save_counter > self.limit_per_file:
                self.iq_save_file.close()
                self.iq_save_file = open(f"{LOG_DIR}/iqs_{int(time.time())}.bin", "ab")

        iq_arr = np.frombuffer(data, np.int16)
        if self.iqPlotterGui:
            self.iq_queue.put(iq_arr)

        if self.demoGui:
            self.demo_queue.put(("iq_data", iq_arr))

        if self.control:
            dapp_logger.debug("Start control operations")
            iq_comp = iq_arr[::2] + iq_arr[1::2] * 1j
            abs_iq = abs(iq_comp).astype(float)
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
                dapp_logger.info(f'abs_iq_av_db: {abs_iq_av_db.mean()}')
                dapp_logger.info(f'abs_iq_av_db_offset_correct: {abs_iq_av_db_offset_correct.mean()}')
                dapp_logger.info(f'--- MAX VALUES ----')
                dapp_logger.info(f'abs_iq_av_db: {abs_iq_av_db.max()}')
                dapp_logger.info(f'abs_iq_av_db_offset_correct: {abs_iq_av_db_offset_correct.max()}')

                # Blacklisting based on the noise floor threshold
                f_ind = np.arange(self.FFT_SIZE)
                blklist_sub_carrier = f_ind[abs_iq_av_db_offset_correct > self.Noise_floor_threshold]
                np.sort(blklist_sub_carrier)
                dapp_logger.info(f'blklist_sub_carrier: {blklist_sub_carrier}')
                prb_blk_list = np.unique((np.floor(blklist_sub_carrier/self.Num_car_prb))).astype(np.uint16)
                dapp_logger.info(f'prb_blk_list: {prb_blk_list}')
                prb_blk_list = prb_blk_list[prb_blk_list > self.prb_thrs]
                dapp_logger.info(f"Blacklisted prbs: {prb_blk_list}")
                prb_new = prb_blk_list.view(prb_blk_list.dtype.newbyteorder('>'))
                self.prb_queue.put(((prb_new, prb_blk_list.size)))

                if self.energyGui:
                    self.sig_queue.put(abs_iq_av_db)
                
                if self.demoGui:
                    self.demo_queue.put(("prb_list", prb_blk_list))

                # reset the variables
                self.abs_iq_av = np.zeros(self.FFT_SIZE)
                self.control_count = 1  

    def prb_update(self, prb_blk_list, n):
        array1 = n.to_bytes(2, "little")
        array2 = prb_blk_list.tobytes(order="C")
        self.prb_updates_socket.send(array1 + array2)

    def prb_update_task(self):
        if not self.control:
            return

        if self.profile:
            self.prb_profiler = cProfile.Profile()
            self.prb_profiler.enable()

        while not self.stop_event.is_set():
            try:
                prb_blk_list, n = self.prb_queue.get(timeout=1.5)
                self.prb_update(prb_blk_list, n)
            except queue.Empty:
                dapp_logger.debug("Empty queue")

        if self.profile:
            self.prb_profiler.disable()
            with open(f"{LOG_DIR}/prb_update.txt", "w") as f:
                p = pstats.Stats(self.prb_profiler, stream=f)
                p.sort_stats("cumtime").print_stats()

    def control_loop(self):
        dapp_logger.debug(f"Start control loop")
        try:
            while not self.stop_event.is_set():
                if self.energyGui:
                    abs_iq_av_db = self.sig_queue.get()
                    self.energyPlotter.process_iq_data(abs_iq_av_db)
                if self.iqPlotterGui:
                    iq_data = self.iq_queue.get()
                    self.iqPlotter.process_iq_data(iq_data)
                if self.demoGui:
                    message = self.demo_queue.get()
                    self.demo.process_iq_data(message)
        except KeyboardInterrupt:
            dapp_logger.error("Keyboard interrupt, closing dApp")
            self.stop_event.set()

    def stop(self):
        dapp_logger.info('Stop of the dApp')
        self.stop_event.set()
        
        if self.control:
            # close connection socket with the client
            self.control_thread.join()
            self.prb_updates_socket.close()
            dapp_logger.info("Connection to client for control closed")

        self.e3_interface.stop_server()
        dapp_logger.info("Stopped server")
        
        if self.save_iqs:
            self.iq_save_file.close()
        
        if self.demoGui:
            self.demo.stop()

def stop_program(dapp):
    dapp.stop()
    print("Test completed.")

def main(args, time: float = 400.0):
    dapp = DApp(ota=args.ota, save_iqs=args.save_iqs, control=args.control, profile=args.profile, energyGui=args.energy_gui, iqPlotterGui=args.iq_plotter_gui, demoGui=args.demo_gui)

    if args.timed:
        timer = threading.Timer(time, stop_program, [dapp])
        timer.start()
    else:
        timer = None

    try:
        dapp.control_loop()
    finally:
        if args.timed:
            timer.cancel()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dApp example")
    parser.add_argument('--ota', action='store_true', default=False, help="Specify if this is OTA or on Colosseum")
    parser.add_argument('--save-iqs', action='store_true', default=False, help="Specify if this is data collection run or not. In the first case I/Q samples will be saved")
    parser.add_argument('--control', action='store_true', default=False, help="Set whether to perform control of PRB")
    parser.add_argument('--energy-gui', action='store_true', default=False, help="Set whether to enable the energy GUI")
    parser.add_argument('--iq-plotter-gui', action='store_true', default=False, help="Set whether to enable the IQ Plotter GUI")
    parser.add_argument('--demo-gui', action='store_true', default=False, help="Set whether to enable the Demo GUI")
    parser.add_argument('--profile', action='store_true', default=False, help="Enable profiling with cProfile")
    parser.add_argument('--timed', action='store_true', default=False, help="Run with a 5-minute time limit")

    args = parser.parse_args()

    if args.profile:
        import cProfile
        import pstats
        cProfile.run('main(args)', 'dapp_profile')

        with open(f"{LOG_DIR}/dapp.txt", "w") as f:
            p = pstats.Stats('dapp_profile', stream=f)
            p.sort_stats('cumtime').print_stats()
        os.remove('dapp_profile')
    else:
        main(args)
