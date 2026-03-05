#!/usr/bin/env python3
"""
Example script to run the minimal SimpleDApp against libe3's simple_agent.

Usage:
    python examples/simple_dapp.py                          # defaults (zmq / ipc)
    python examples/simple_dapp.py --link zmq --transport tcp
    python examples/simple_dapp.py --control                # echo control actions back
    python examples/simple_dapp.py --timed 30               # run for 30 seconds then stop
"""

import argparse
import threading
import time

from e3interface.e3_connector import E3LinkLayer, E3TransportLayer
from simple.simple_dapp import SimpleDApp


def stop_program(timeout_sec: int, dapp: SimpleDApp):
    """Background thread that stops the dApp after *timeout_sec* seconds."""
    time.sleep(timeout_sec)
    print(f"[INFO] Timer elapsed after {timeout_sec} seconds")
    dapp.stop_event.set()
    time.sleep(0.5)
    print("[INFO] Stopping of the dApp completed")


def main(args):
    dapp = SimpleDApp(
        link=args.link,
        transport=args.transport,
        encoding_method=args.encoding,
        control=args.control,
    )

    # ---- E3 Setup -------------------------------------------------------- #
    response, setup_response = dapp.setup_connection()
    if not response:
        raise RuntimeError("[ERROR] RAN refused Setup or a connection error happened")

    ran_functions = setup_response["ranFunctionList"]
    print(f"[INFO] Setup complete — RAN functions available: {ran_functions}")

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
                    f"[INFO] Decoded ranFunctionData for RAN function {ran_function['ranFunctionIdentifier']}: {decoded}"
                )

    time.sleep(1)

    # ---- Subscription ---------------------------------------------------- #
    dapp.send_subscription_request()

    # ---- Optional timer -------------------------------------------------- #
    timer = None
    if args.timed:
        timer = threading.Thread(
            target=stop_program, args=(args.timed, dapp), daemon=False
        )
        timer.start()

    # ---- Main loop ------------------------------------------------------- #
    try:
        dapp.control_loop()
    finally:
        if timer is not None and timer.is_alive():
            timer.join(timeout=2)
            if timer.is_alive():
                print("[ERROR] Timer thread did not terminate in time")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Minimal example dApp for the Simple service model (libe3 simple_agent)"
    )
    parser.add_argument(
        "--link",
        type=str,
        default="zmq",
        choices=[layer.value for layer in E3LinkLayer],
        help="Link layer (default: zmq)",
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="ipc",
        choices=[layer.value for layer in E3TransportLayer],
        help="Transport layer (default: ipc)",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="asn1",
        choices=["asn1", "json"],
        help="Encoding format (default: asn1)",
    )
    parser.add_argument(
        "--control",
        action="store_true",
        default=False,
        help="Echo a control action back to the agent for each indication",
    )
    parser.add_argument(
        "--timed",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Run for N seconds then stop (0 = unlimited)",
    )

    args = parser.parse_args()
    print("Start SimpleDApp")
    main(args)
