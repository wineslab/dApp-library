#!/usr/bin/env python3
"""
Example script to showcase the Nvidia dApp 
"""

import argparse
import time

from e3interface.e3_connector import E3LinkLayer, E3TransportLayer
from nvidia.nvidia_dapp import NvidiaDApp


def main(args):
    dapp = NvidiaDApp(dapp_name="NvidiaKPMs", dapp_version="1.0.0", vendor="WinesLab",
                      link=args.link, transport=args.transport, encoding_method="json")
    response, setup_response = dapp.setup_connection()
    
    if not response:
        raise ValueError("[WARNING] RAN refused Setup")
    
    ran_functions = setup_response["ranFunctionList"]

    print(f"[INFO] Setup Complete - RAN function available: {len(ran_functions)}")
    time.sleep(1)
    # atm we subscribe to all

    dapp.send_subscription_request(ran_functions)
    
    try:
        dapp.control_loop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example of a dApp for Nvidia data extraction")
    parser.add_argument('--link', type=str, default='zmq', choices=[layer.value for layer in E3LinkLayer], help="Specify the link layer to be used")
    parser.add_argument('--transport', type=str, default='tcp', choices=[layer.value for layer in E3TransportLayer], help="Specify the transport layer to be used")

    args = parser.parse_args()
    print("Start dApp")

    main(args)
