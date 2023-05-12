#!/usr/bin/env python3
"""
Module Docstring
"""

__author__ = "Andrea Lacava"
__version__ = "0.1.0"
__license__ = "MIT"

import socket
import argparse
from typing_extensions import override

from dapp.dapp import DApp


def extract_iq_samples(hostname: str, port: int):
    """ Main entry point of the app """
    protoname = "tcp"
    server_hostname = hostname if hostname else "127.0.0.1"
    server_port = port if port else 12346
    sockfd : socket.socket

    print(f'{server_hostname}:{server_port}')

    try:
        # Get socket
        protoent = socket.getprotobyname(protoname)
        sockfd = socket.socket(socket.AF_INET, socket.SOCK_STREAM, protoent)

        # Prepare sockaddr_in
        sockaddr_in = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sockaddr_in.connect((server_hostname, server_port))

        while True:
            print("enter integer (1 dump, 0 stop, empty to quit):")
            user_input = input()

            if not user_input:
                break

            sockfd.sendall(user_input.encode())

            # # Read response from the server
            # while True:
            #     data = sockfd.recv(1024)
            #     if not data:
            #         break
            #     buffer.extend(data)
            #     if buffer[-1] == b'\n':
            #         sys.stdout.buffer.write(buffer)
            #         sys.stdout.flush()
            #         buffer.clear()
            #         break

    except socket.error as e:
        print("Socket error:", e)
    except KeyboardInterrupt:
        print("Program terminated.")
    finally:
        if sockfd:
            sockfd.close()

class SpearApp(DApp):
    @override
    def method_to_override(self):
        print("Overridden method")


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--server", "--hostname", action="store", dest="hostname")
    parser.add_argument("-p", "--port", action="store", dest="port")

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__))

    args = parser.parse_args()

    extract_iq_samples(args.hostname, args.port)
