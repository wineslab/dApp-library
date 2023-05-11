#!/usr/bin/env python3
"""
Module Docstring
"""

__author__ = "Your Name"
__version__ = "0.1.0"
__license__ = "MIT"

import socket
import sys
import argparse


def extract_iq_samples(hostname: str = None, port: int = None):
    """ Main entry point of the app """
    protoname = "tcp"
    server_hostname = hostname if not hostname else "127.0.0.1"
    server_port = port if not port else 12346

    print(server_hostname, ':', server_port)

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
