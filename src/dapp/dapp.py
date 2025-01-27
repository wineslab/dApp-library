#!/usr/bin/env python3
"""
Example of a dApp
"""

__author__ = "Andrea Lacava"
__version__ = "0.1.0"
__license__ = "MIT"

from abc import ABC
import argparse
import multiprocessing
import time
import sys
import numpy as np
# np.set_printoptions(threshold=sys.maxsize)
import e3_interface, e3_logging # used for profiling
from e3_connector import E3LinkLayer, E3TransportLayer
from e3_interface import E3Interface
from e3_logging import dapp_logger
import os
import yappi


LOG_DIR = ('.' if os.geteuid() != 0 else '') + '/logs/'

class DApp(ABC):
    DAPP_ID = 1
    e3_interface: E3Interface

    ###  Configuration ###
    # gNB runs with BW = 40 MHz, with -E (3/4 sampling)
    # Center frequency = 3.6192 GHz
    # OAI sampling frequency 46.08e6
    # No SRS for now, hence in gNB config file set do_SRS = 0
    # gNB->frame_parms.ofdm_symbol_size = 1536
    # gNB->frame_parms.first_carrier_offset = 900
    # Noise floor threshold needs to be calibrated
    # We receive the symbols and average them over some frames, and do thresholding.

    def __init__(self, ota: bool = False, save_iqs: bool = False, control: bool = False, link: str = 'posix', transport:str = 'uds', **kwargs):
        super().__init__()
        self.profile = kwargs.get('profile', False)
        self.e3_interface = E3Interface(link=link, transport=transport, profile=self.profile)
        self.stop_event = multiprocessing.Event()

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

        dapp_logger.info(f'Using {link} and {transport}')

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
            self.sig_queue = multiprocessing.Queue() 
            self.energyPlotter = EnergyPlotter(self.FFT_SIZE, bw=self.bw, center_freq=self.center_freq) 

        if self.iqPlotterGui:
            from iq_plotter import IQPlotter
            self.iq_queue = multiprocessing.Queue() 
            iq_size = self.FFT_SIZE * 2 # double the size of ofdm_symbol_size since real and imaginary parts are interleaved
            self.iqPlotter = IQPlotter(buffer_size=500, iq_size=iq_size, bw=self.bw, center_freq=self.center_freq)    

        if self.demoGui:
            from demo_plotter import DemoGui
            self.demo_queue = multiprocessing.Queue()
            iq_size = self.FFT_SIZE * 2 # double the size of ofdm_symbol_size since real and imaginary parts are interleaved
            self.demo = DemoGui(buffer_size=100, iq_size=iq_size) 

    def setup_connection(self):
        while True:    
            response = self.e3_interface.send_setup_request(self.DAPP_ID)
            dapp_logger.info(f'E3 Setup Response: {response}')
            if response:
               break
            dapp_logger.warning('RAN refused setup or dApp was not able to connect, waiting 2 secs')
            time.sleep(2)

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


                # Blacklisting based on the noise floor threshold
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
            self.stop()

    def stop(self):
        dapp_logger.info('Stop of the dApp')
        self.stop_event.set()

        self.e3_interface.terminate_connections()
        dapp_logger.info("Stopped server")
        
        if self.save_iqs:
            self.iq_save_file.close()
        
        if self.demoGui:
            self.demo.stop()

def stop_program(time_to_wait, dapp: DApp):
    time.sleep(time_to_wait)
    print("Stop is called")
    dapp.stop()
    time.sleep(0.5) # to allow proper closure of the dApp threads, irrelevant to profiling
    print("Test completed")

def main(args, time_to_wait: float = 60.0):
    # with open(f"{LOG_DIR}/busy.txt", "w") as f:
    #     f.close()
    
    dapp = DApp(ota=args.ota, save_iqs=args.save_iqs, control=args.control, profile=args.profile, 
                link=args.link, transport=args.transport,
                energyGui=args.energy_gui, iqPlotterGui=args.iq_plotter_gui, demoGui=args.demo_gui)

    dapp.setup_connection()
    
    if args.timed:
        timer = multiprocessing.Process(target=stop_program, args=(time_to_wait, dapp))
        timer.start()
    else:
        timer = None

    try:
        dapp.control_loop()
    finally:
        if args.timed:
            timer.kill()

    if args.profile:
        time.sleep(2.5)

        with open(f"{LOG_DIR}/busy.txt", "r") as file:
            numbers = [float(line.strip()) for line in file]
        average_busy_wait = sum(numbers) / len(numbers)

        with open("f{LOG_DIR}/func_stats_inbound.txt", "r") as file:
            lines = file.readlines()
            print(lines[4].strip())
            receive_msg = lines[5].strip()
            handle = lines[6].strip()

        with open("f{LOG_DIR}/func_stats_outbound.txt", "r") as file:
            lines = file.readlines()
            control_act = lines[5].strip()
            send_msg = lines[6].strip()

        print(receive_msg)
        print(f"Average busy wait: {average_busy_wait:>30}")
        print(handle)
        print(control_act)
        print(send_msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dApp example")
    parser.add_argument('--ota', action='store_true', default=False, help="Specify if this is OTA or on Colosseum")
    parser.add_argument('--link', type=str, default='posix', choices=[layer.value for layer in E3LinkLayer], help="Specify the link layer to be used")
    parser.add_argument('--transport', type=str,  default='uds', choices=[layer.value for layer in E3TransportLayer], help="Specify the transport layer to be used")
    parser.add_argument('--save-iqs', action='store_true', default=False, help="Specify if this is data collection run or not. In the first case I/Q samples will be saved")
    parser.add_argument('--control', action='store_true', default=False, help="Set whether to perform control of PRB")
    parser.add_argument('--energy-gui', action='store_true', default=False, help="Set whether to enable the energy GUI")
    parser.add_argument('--iq-plotter-gui', action='store_true', default=False, help="Set whether to enable the IQ Plotter GUI")
    parser.add_argument('--demo-gui', action='store_true', default=False, help="Set whether to enable the Demo GUI")
    parser.add_argument('--profile', action='store_true', default=False, help="Enable profiling with cProfile")
    parser.add_argument('--timed', action='store_true', default=False, help="Run with a 5-minute time limit")

    args = parser.parse_args()

    print("Start dApp")

    if args.profile:
        yappi.set_clock_type("wall")
        yappi.start() 
        main(args)

        current_module = sys.modules[__name__]
        with open(f"{LOG_DIR}/func_stats_dapp.txt", "w") as f:
            yappi.get_func_stats(
                filter_callback=lambda x: yappi.module_matches(
                    x, [current_module, e3_interface, e3_logging]
                )
            ).strip_dirs().sort("ncall", sort_order="desc").print_all(
                f,
                columns={
                    0: ("name", 60),
                    1: ("ncall", 10),
                    2: ("tsub", 8),
                    3: ("ttot", 8),
                    4: ("tavg", 8),
                },
            )
    else:
        main(args)
