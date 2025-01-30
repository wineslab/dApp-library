#!/usr/bin/env python3
"""
Usage example for the Spectrum Sharing dApp 
"""

import argparse
import multiprocessing
import os
import time

from e3interface.e3_connector import E3LinkLayer, E3TransportLayer
from spectrum.spectrum_dapp import SpectrumSharingDApp

LOG_DIR = ('.' if os.geteuid() != 0 else '') + '/logs/'

def stop_program(time_to_wait, dapp: SpectrumSharingDApp):
    time.sleep(time_to_wait)
    print("Stop is called")
    dapp.stop()
    time.sleep(0.5) # to allow proper closure of the dApp threads, irrelevant to profiling
    print("Test completed")

def main(args, time_to_wait: float = 60.0):
    # with open(f"{LOG_DIR}/busy.txt", "w") as f:
    #     f.close()
    
    dapp = SpectrumSharingDApp(ota=args.ota, save_iqs=args.save_iqs, control=args.control, link=args.link, transport=args.transport,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dApp example")
    parser.add_argument('--ota', action='store_true', default=False, help="Specify if this is OTA or on Colosseum")
    parser.add_argument('--link', type=str, default='zmq', choices=[layer.value for layer in E3LinkLayer], help="Specify the link layer to be used")
    parser.add_argument('--transport', type=str,  default='ipc', choices=[layer.value for layer in E3TransportLayer], help="Specify the transport layer to be used")
    parser.add_argument('--save-iqs', action='store_true', default=False, help="Specify if this is data collection run or not. In the first case I/Q samples will be saved")
    parser.add_argument('--control', action='store_true', default=False, help="Set whether to perform control of PRB")
    parser.add_argument('--energy-gui', action='store_true', default=False, help="Set whether to enable the energy GUI")
    parser.add_argument('--iq-plotter-gui', action='store_true', default=False, help="Set whether to enable the IQ Plotter GUI")
    parser.add_argument('--demo-gui', action='store_true', default=False, help="Set whether to enable the Demo GUI")
    parser.add_argument('--timed', action='store_true', default=False, help="Run with a 5-minute time limit")

    args = parser.parse_args()

    print("Start dApp")

    main(args)
