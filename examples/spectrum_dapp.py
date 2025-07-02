#!/usr/bin/env python3
"""
Example script to showcase the Spectrum Sharing dApp 
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

    if args.model:
        try:
            from libiq.classifier.cnn import Classifier
        except ModuleNotFoundError:
            print(
                "Optional dependencies to run this example are not installed.\n"
                "Fix this by running:\n\n"
                "    pip install libiq  # OR\n"
                "    pip install dapps[cnn] (minimal) # OR\n"
                "    pip install dapps[all] (preferred)\n"
            )
            exit(-1)

    # This value really depends on the RF conditions and the RU used and should be carefully calibrated
    if args.noise_floor_threshold:
        print(f'Using custom configuration')
        noise_floor_threshold = args.noise_floor_threshold
    else:    
        if args.ota:
            print(f'Using OTA configuration')
            noise_floor_threshold = 20 
        else: # Colosseum
            print(f'Using Colosseum configuration')
            noise_floor_threshold = 53

    print(f'Threshold is {noise_floor_threshold}')

    classifier = None
    if args.model:
        import math
        ofdm_symbol_size = args.num_prbs * 12  # 12 subcarriers per PRB always fixed by LTE/NR standard
        fft_size = 2 ** math.ceil(math.log2(ofdm_symbol_size))  # Next power of 2
        classifier = Classifier(
            time_window = args.time_window,
            input_vector = fft_size,
            moving_avg_window = args.moving_avg_window,
            extraction_window = args.extraction_window,
            model_path = args.model
        )

    dapp = SpectrumSharingDApp(noise_floor_threshold=noise_floor_threshold, save_iqs=args.save_iqs, control=args.control, link=args.link, transport=args.transport,
                energyGui=args.energy_gui, iqPlotterGui=args.iq_plotter_gui, dashboard=args.demo_gui, classifier=classifier, center_freq=args.center_freq,
                num_prbs=args.num_prbs, e_sampling=args.e, num_subcarrier_spacing= args.num_subcarrier_spacing)

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
    parser = argparse.ArgumentParser(description="Example of a dApp for Spectrum Sharing")
    parser.add_argument('--link', type=str, default='zmq', choices=[layer.value for layer in E3LinkLayer], help="Specify the link layer to be used")
    parser.add_argument('--transport', type=str, default='ipc', choices=[layer.value for layer in E3TransportLayer], help="Specify the transport layer to be used")
    parser.add_argument('--save-iqs', action='store_true', default=False, help="Specify if this is data collection run or not. In the first case I/Q samples will be saved")
    parser.add_argument('--control', action='store_true', default=False, help="Set whether to perform control of PRB")
    parser.add_argument('--noise-floor-threshold', type=int, default=None, help="Set the noise floor threshold for determining the presence of incumbents and for detecting the PRBs affected.")
    parser.add_argument('--ota', action='store_true', default=False, help="Specify if the setup used is OTA or on Colosseum for determining the noise floor threshold. If the `noise_floor_threshold` parameter is specified, this parameter is ignored")
    parser.add_argument('--energy-gui', action='store_true', default=False, help="Set whether to enable the energy GUI")
    parser.add_argument('--iq-plotter-gui', action='store_true', default=False, help="Set whether to enable the IQ Plotter GUI")
    parser.add_argument('--demo-gui', action='store_true', default=False, help="Set whether to enable the Demo GUI")
    parser.add_argument('--num-prbs', type=int, default=106, help="Number of PRBs")
    parser.add_argument('--num-subcarrier-spacing', type=int, default=30, help="Subcarrier spacing in kHz (FR1 is 30)")
    parser.add_argument('--e', action='store_true', default=False, help="Set if 3/4 sampling for FFT size is set on the gNB (-E option on OAI)")
    parser.add_argument('--center-freq', type=float, default=3.6192e9, help="Center frequency in Hz")
    parser.add_argument('--timed', action='store_true', default=False, help="Run with a 5-minute time limit")
    parser.add_argument('--model', type=str, default='', help="Path to the CNN model file to be used")
    parser.add_argument('--time-window', type=int, default=5, help="Number of input vectors to pass to the CNN model.")
    parser.add_argument('--moving-avg-window', type=int, default=30, help="Window size (in samples) for the moving average used to detect energy peaks in the spectrum.")
    parser.add_argument('--extraction-window', type=int, default=600, help="Number of samples to retain after detecting an energy peak.")

    args = parser.parse_args()
    print("Start dApp")

    main(args)
