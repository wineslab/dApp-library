#!/usr/bin/env python3
"""
Example script to showcase the Spectrum Sharing dApp
"""

import argparse
import threading
import time
import logging

from e3interface.e3_connector import E3LinkLayer, E3TransportLayer
from spectrum.spectrum_dapp import SpectrumSharingDApp, compute_fft_size
from spectrum.threshold_detector import StaticThresholdDetector, AdaptiveThresholdDetector

LOG_DIR = '/tmp/'

def stop_program(time_to_wait, dapp: SpectrumSharingDApp):
    time.sleep(time_to_wait)
    print(f"[INFO] Timer elapsed after {time_to_wait} seconds")
    dapp.stop_event.set()
    time.sleep(0.5) # to allow proper closure of the dApp threads, irrelevant to profiling
    print("[INFO] Stopping of the dApp completed")

def main(args):
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

    # This value really depends on the RF conditions and the RU used and
    # should be carefully calibrated.
    if args.noise_floor_threshold:
        print('Using custom configuration')
        noise_floor_threshold = args.noise_floor_threshold
    else:
        if args.ota:
            print('Using OTA configuration')
            noise_floor_threshold = 20
        else:  # Colosseum
            print('Using Colosseum configuration')
            noise_floor_threshold = 53

    print(f'Threshold is {noise_floor_threshold}')

    # ------------------------------------------------------------------
    # Build the detection strategy explicitly so the example is the
    # authoritative place for detector configuration.
    # ------------------------------------------------------------------
    fft_size = compute_fft_size(args.num_prbs, args.e)

    if args.use_adaptive_noise_floor:
        detector = AdaptiveThresholdDetector(
            snr_threshold_db=noise_floor_threshold,
            fft_size=fft_size,
            hist_depth=args.average_over_frames,
            embargo_timeout_secs=args.embargo_timeout_secs,
        )
        print(
            f"[INFO] Detector: AdaptiveThresholdDetector"
            f" | SNR threshold: {noise_floor_threshold} dB"
            f" | hist_depth: {args.average_over_frames}"
            f" | embargo: {args.embargo_timeout_secs} s"
        )
    else:
        detector = StaticThresholdDetector(
            threshold_db=noise_floor_threshold,
            fft_size=fft_size,
            window=args.average_over_frames,
        )
        print(
            f"[INFO] Detector: StaticThresholdDetector"
            f" | threshold: {noise_floor_threshold} dB"
            f" | window: {args.average_over_frames} frames"
        )

    classifier = None
    if args.model:
        classifier = Classifier(
            time_window=args.time_window,
            input_vector=fft_size,
            moving_avg_window=args.moving_avg_window,
            extraction_window=args.extraction_window,
            model_path=args.model,
        )

    dapp = SpectrumSharingDApp(
        detector=detector,
        save_iqs=args.save_iqs,
        control=args.control,
        link=args.link,
        transport=args.transport,
        energyGui=args.energy_gui,
        iqPlotterGui=args.iq_plotter_gui,
        dashboard=args.demo_gui,
        classifier=classifier,
        center_freq=args.center_freq,
        num_prbs=args.num_prbs,
        e_sampling=args.e,
        num_subcarrier_spacing=args.num_subcarrier_spacing,
        sampling_threshold=args.sampling_threshold,
        dapp_name="SpectrumSharing",
        dapp_version="1.0.0",
        vendor="WinesLab",
    )

    response, setup_response = dapp.setup_connection()

    if not response:
        raise ValueError("[WARNING] RAN refused Setup")

    ran_functions = setup_response["ranFunctionList"]
    print(f"[INFO] Setup Complete - RAN function available: {ran_functions}")

    for ran_function in ran_functions:
        if dapp.check_sm_ids(
            ran_function["ranFunctionIdentifier"],
            ran_function["telemetryIdentifierList"],
            ran_function["controlIdentifierList"],
        ):
            # Attempt to decode ranFunctionData if present
            rfd = ran_function.get("ranFunctionData")
            if rfd:
                decoded = dapp.decode_ran_function_data(rfd)
                print(
                    f"[INFO] Decoded ranFunctionData for RAN function"
                    f" {ran_function['ranFunctionIdentifier']}: {decoded}"
                )
    time.sleep(1)

    dapp.send_subscription_request()

    if args.timed:
        timer = threading.Thread(target=stop_program, args=(args.timed, dapp), daemon=False)
        timer.start()
    else:
        timer = None

    try:
        dapp.control_loop()
    finally:
        if args.timed and timer is not None:
            if timer.is_alive():
                timer.join(timeout=2)
                if timer.is_alive():
                    print("[ERROR] Timer thread did not terminate in time")

    logging.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example of a dApp for Spectrum Sharing")
    parser.add_argument('--link', type=str, default='zmq',
                        choices=[layer.value for layer in E3LinkLayer],
                        help="Link layer to use")
    parser.add_argument('--transport', type=str, default='ipc',
                        choices=[layer.value for layer in E3TransportLayer],
                        help="Transport layer to use")
    parser.add_argument('--save-iqs', action='store_true', default=False,
                        help="Save I/Q samples to SigMF files")
    parser.add_argument('--control', action='store_true', default=False,
                        help="Send PRB blacklist control messages to the gNB")
    parser.add_argument('--noise-floor-threshold', type=int, default=None,
                        help="Detection threshold in dB (static) or dB above noise floor (adaptive)")
    parser.add_argument('--use-adaptive-noise-floor', action='store_true', default=False,
                        help="Use per-bin median noise floor estimation instead of a fixed threshold")
    parser.add_argument('--embargo-timeout-secs', type=float, default=10.1,
                        help="Hold time in seconds for embargoed PRBs after last detection (adaptive mode)")
    parser.add_argument('--average-over-frames', type=int, default=64,
                        help="Number of frames to average before each decision")
    parser.add_argument('--ota', action='store_true', default=False,
                        help="Use OTA threshold (20 dB) instead of Colosseum (53 dB). "
                             "Ignored when --noise-floor-threshold is set.")
    parser.add_argument('--energy-gui', action='store_true', default=False,
                        help="Enable energy spectrum visualization")
    parser.add_argument('--iq-plotter-gui', action='store_true', default=False,
                        help="Enable IQ time-domain plotter")
    parser.add_argument('--demo-gui', action='store_true', default=False,
                        help="Enable dashboard visualization")
    parser.add_argument('--num-prbs', type=int, default=106,
                        help="Number of PRBs")
    parser.add_argument('--num-subcarrier-spacing', type=int, default=30,
                        help="Subcarrier spacing in kHz (FR1 = 30)")
    parser.add_argument('--e', action='store_true', default=False,
                        help="Enable 3/4 FFT sampling (OAI -E flag for USRPs)")
    parser.add_argument('--center-freq', type=float, default=3.6192e9,
                        help="RF center frequency in Hz")
    parser.add_argument('--timed', type=int, default=0, metavar='SECONDS',
                        help="Stop automatically after SECONDS (0 = run indefinitely)")
    parser.add_argument('--model', type=str, default='',
                        help="Path to CNN model file for signal classification")
    parser.add_argument('--time-window', type=int, default=5,
                        help="Input vector count for CNN model")
    parser.add_argument('--moving-avg-window', type=int, default=30,
                        help="Moving average window for CNN energy peak detection")
    parser.add_argument('--extraction-window', type=int, default=600,
                        help="Samples to retain after CNN energy peak detection")
    parser.add_argument('--sampling-threshold', type=int, default=5,
                        help="Deliver IQ every N sensing cycles (each cycle is 10 ms)")

    args = parser.parse_args()
    print("Start dApp")

    main(args)
