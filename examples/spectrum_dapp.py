#!/usr/bin/env python3
"""
Example script to showcase the Spectrum Sharing dApp
"""

import argparse
import threading
import time
import logging

from e3interface.e3_connector import E3LinkLayer, E3TransportLayer
from spectrum.spectrum_dapp import (
    SpectrumSharingDApp,
    compute_fft_size,
    make_periodic_toggle_callback,
)
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
        # CNN signal classification is not wired into SpectrumSharingDApp; a
        # classifier= kwarg would fall through to **kwargs and be dropped, so
        # --model (and --time-window/--moving-avg-window/--extraction-window)
        # would silently do nothing. Fail loudly instead.
        raise SystemExit(
            "--model (CNN signal classification) is not supported by "
            "SpectrumSharingDApp; remove --model and the associated "
            "--time-window/--moving-avg-window/--extraction-window flags."
        )

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

    dapp = SpectrumSharingDApp(
        detector=detector,
        save_iqs=args.save_iqs,
        control=args.control,
        link=args.link,
        transport=args.transport,
        energyGui=args.energy_gui,
        iqPlotterGui=args.iq_plotter_gui,
        dashboard=args.demo_gui,
        viz_web_port=args.viz_web_port,
        viz_zmq_port=args.viz_zmq_port,
        external_viz=args.external_viz,
        center_freq=args.center_freq,
        num_prbs=args.num_prbs,
        e_sampling=args.e,
        num_subcarrier_spacing=args.num_subcarrier_spacing,
        sampling_threshold=args.sampling_threshold,
        max_samples_per_file=args.max_samples_per_file,
        fp16_beta=args.fp16_beta,
        sensing_only=args.sensing_only,
        strict_sensing=args.strict_sensing,
        min_sensing_symbols=args.min_sensing_symbols,
        encoding_method=args.encoding_method,
        ground_truth=args.ground_truth,
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

    # Optional sensing-policy toggle.  When --toggle-period > 0, installs
    # a periodic on/off toggle that flips the gNB's masked UL TDA
    # selector via spectrum_sm ctrl_id=2.  Operator must have configured
    # additional_ul_tdas in the gNB (e.g. "0:7") for the scheduler to
    # have a short TDA to switch to.  See examples/README or the gNB
    # boot log ([additional] tag) to verify.
    if args.toggle_period > 0:
        print(f"[INFO] Installing sensing-policy toggle: "
              f"period={args.toggle_period}s "
              f"n_slots={args.toggle_n_slots} "
              f"mask_when_on=0x{args.toggle_mask:04x}")
        dapp.set_sensing_policy_logic(make_periodic_toggle_callback(
            period_s=args.toggle_period,
            n_slots=args.toggle_n_slots,
            mask_when_on=args.toggle_mask,
        ))

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
                        help="Send PRB-block control messages to the gNB when "
                             "PRBs are detected above the noise threshold")
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
                        help="Publish per-slot subcarrier power to the visualizer")
    parser.add_argument('--viz-web-port', type=int, default=5001,
                        help="Visualizer web UI port (default 5001)")
    parser.add_argument('--viz-zmq-port', type=int, default=5559,
                        help="ZMQ port the dApp publishes on / the visualizer reads (default 5559)")
    parser.add_argument('--external-viz', action='store_true', default=False,
                        help="Don't spawn the visualizer; publish to an already-running one")
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
                        help="Render a new dashboard frame every N IQ batches (visualization only, does not affect IQ delivery or recording)")
    parser.add_argument('--max-samples-per-file', type=int, default=46_080_000,
                        help="Rotate the SigMF capture file once a segment reaches "
                             "this many true IQ samples (default 46080000). Each "
                             "indication is kept whole, so a segment may exceed the "
                             "threshold by up to one indication's worth of samples. "
                             "Segment wall-clock duration depends on the capture rate. "
                             "Only used with --save-iqs.")
    parser.add_argument('--ground-truth', type=str, default='', metavar='LABEL',
                        help="Initial ground truth label written into IQ annotations "
                             "(only used with --save-iqs). "
                             "Updatable at runtime via the dashboard GUI "
                             "when --demo-gui is also set.")
    parser.add_argument('--fp16-beta', type=float, default=1.0 / 2048.0,
                        help="FP16 IQ rescale factor; MUST match the gNB "
                             "E3Configuration.fp16_beta (the reader rescales by "
                             "1/beta). Default 1/2048 matches the gNB code default; "
                             "the X410 sample conf overrides it to 0.0078125 (1/128).")
    parser.add_argument('--encoding-method', type=str, default='asn1',
                        choices=['asn1', 'json'],
                        help="Wire encoding for Spectrum-* envelopes (default: asn1).")
    parser.add_argument('--no-sensing-only', dest='sensing_only',
                        action='store_false', default=True,
                        help="Disable the sensing-window filter. By default the dApp uses the "
                             "sensing ranges from the Spectrum SM (RF=1) to slice the detector "
                             "input to sensing-PUSCH cells; passing this flag uses every cell.")
    parser.add_argument('--strict-sensing', dest='strict_sensing',
                        action='store_true', default=False,
                        help="Stricter sensing filter: drop any slot whose sensing window "
                             "doesn't cover the full slot (i.e. any UE PUSCH was granted "
                             "in that slot). Eliminates UE CP/spectral bleed at the cost "
                             "of a sparser waterfall under heavy UE traffic. Requires "
                             "--sensing-only (the default).")
    parser.add_argument('--min-sensing-symbols', type=int, default=1,
                        help="Minimum number of kept symbols required to emit the slot to "
                             "the dashboard. Default 1: display every slot that has any "
                             "sensing symbol — keeps the waterfall scrolling even under "
                             "heavy UE UL where the MAC scheduler leaves only one free "
                             "symbol per slot. Set to 2 to drop the typical \"only sym 13 "
                             "survived\" partial-sensing slots (evens out brightness at "
                             "the cost of dashboard freezes under load). Set to 14 to "
                             "require fully-clean slots (equivalent to "
                             "--strict-sensing).")

    # --- Optional sensing-policy toggle ----------------------------- #
    # Drives the gNB's masked UL TDA selector via spectrum_sm ctrl_id=2.
    # Disabled by default (toggle-period=0); set --toggle-period to a
    # positive value to enable.  Requires additional_ul_tdas (e.g. "0:7")
    # on the gNB so the scheduler has a short TDA to switch to.
    parser.add_argument('--toggle-period', type=float, default=0.0,
                        metavar='SECONDS',
                        help="Toggle the gNB sensing policy on/off every N seconds "
                             "(0 = disabled, default).  When > 0, alternates between "
                             "installing the mask (active) and clearing it (deactivate).  "
                             "Requires additional_ul_tdas configured on the gNB.")
    parser.add_argument('--toggle-n-slots', type=int, default=20,
                        help="numb_slots_frame of the gNB for the per-slot mask "
                             "(mu=1 -> 20).  Must match the gNB exactly or the gNB "
                             "will NACK the policy with an n_slots mismatch.")
    parser.add_argument('--toggle-mask', type=lambda s: int(s, 0), default=0x3F80,
                        metavar='HEX',
                        help="14-bit symbol bitmap applied uniformly to every slot "
                             "when the toggle is active (default 0x3F80 = syms 7..13).")

    args = parser.parse_args()
    print("Start dApp")

    main(args)
