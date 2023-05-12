#!/usr/bin/env python3
"""
Module Docstring
"""

__author__ = "Andrea Lacava"
__version__ = "0.1.0"
__license__ = ""

import sys
sys.path.insert(0, '/home/wineslab/spear-dApp/dapp/')

import socket
import argparse
from typing_extensions import override

from dapp.dapp import DApp

class SpearApp(DApp):
    protoname = "tcp"
    server_hostname : str
    server_port : int
    sockfd : socket.socket

    def __init__(self, **kwargs):
        super().__init__()
        self.server_hostname = kwargs.get('hostname', "127.0.0.1")
        self.server_port = kwargs.get('port', 12346)

    @override
    def method_to_override(self):
        print("Overridden method")

    def extract_iq_samples(self):
        """ Main entry point of the app """
        
        print(f'{self.server_hostname}:{self.server_port}')

        try:
            # Create socket
            protoent = socket.getprotobyname(self.protoname)
            self.sockfd = socket.socket(socket.AF_INET, socket.SOCK_STREAM, protoent)
            self.sockfd.connect((self.server_hostname, self.server_port))

            while True:
                print("enter integer representing the number of iq samples:")
                user_input = input()

                if not user_input:
                    break

                self.sockfd.send(user_input.encode())

                # # Read response from the server
                # while True:
                #     data = self.sockfd.recv(1024)
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
            if self.sockfd:
                self.sockfd.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--server", "--hostname", action="store", dest="hostname")
    parser.add_argument("-p", "--port", action="store", dest="port")

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__))

    args = parser.parse_args()

    spearDApp = SpearApp()

    spearDApp.extract_iq_samples()
