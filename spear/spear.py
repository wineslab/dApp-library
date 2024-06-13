#!/usr/bin/env python3
"""
Module Docstring
"""

__author__ = "Andrea Lacava"
__version__ = "0.1.0"
__license__ = ""

import os
import sys
import time
sys.path.insert(0, '/home/wineslab/spear-dApp/dapp/')

import socket
from typing_extensions import override

from dapp.dapp import DApp
import threading

from .eval_dl_pytorch import evaluate_iq_samples


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

    def run(self):
        # Create two thread objects
        thread1 = threading.Thread(target=self.extract_iq_samples)
        thread2 = threading.Thread(target=self.evaluate_samples)

        # Start both threads
        thread1.start()
        thread2.start()

        # Wait for both threads to finish
        thread1.join()
        thread2.join()
        print('End of the work')

    def evaluate_samples(self):
        """ Evaluation of iq samples """
        file_path = '/home/wineslab/openairinterface5g/iqs_dump/iqs.txt'
        try:
            while True:
                if os.path.exists(file_path):
                    print('Evaluate samples')
                    y_pred, explained = evaluate_iq_samples(input=file_path)
                    print(y_pred)
                    print(explained)
                    
                    print('Remove file')
                    os.remove(file_path)
                else:
                    print("File does not exist. Sleeping for 5 seconds...")
                    time.sleep(5)
        except KeyboardInterrupt:
            print("Program terminated.")


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


