#!/usr/bin/env python3
"""
dApp Base class that should be extended.
This class implements the E3AP and is a wrapper for the access to the E3Interface class.+
"""

__author__ = "Andrea Lacava"

from abc import ABC, abstractmethod
import multiprocessing
import time
from e3interface.e3_interface import E3Interface
from e3interface.e3_logging import dapp_logger
import os


LOG_DIR = ('.' if os.geteuid() != 0 else '') + '/logs/'

class DApp(ABC):
    e3_interface: E3Interface

    def __init__(self, id: int = 1, link: str = 'posix', transport:str = 'uds', callbacks: list = [], **kwargs):
        super().__init__()
        self.dapp_id = id
        self.e3_interface = E3Interface(link=link, transport=transport)
        self.stop_event = multiprocessing.Event()

        dapp_logger.info(f'Using {link} and {transport}')

        for callback in callbacks:
            self.e3_interface.add_callback(callback)

    def setup_connection(self):
        while True:    
            response = self.e3_interface.send_setup_request(self.dapp_id)
            dapp_logger.info(f'E3 Setup Response: {response}')
            if response:
               break
            dapp_logger.warning('RAN refused setup or dApp was not able to connect, waiting 2 secs')
            time.sleep(2)

    @abstractmethod
    def _control_loop(self):
        pass

    def control_loop(self):
        dapp_logger.debug(f"Start control loop")
        try:
            while not self.stop_event.is_set():
                self._control_loop()
        except KeyboardInterrupt:
            dapp_logger.error("Keyboard interrupt, closing dApp")
            self.stop()

    @abstractmethod
    def _stop(self):
        pass

    def stop(self):
        dapp_logger.info('Stop of the dApp')
        self.stop_event.set()

        self.e3_interface.terminate_connections()
        dapp_logger.info("Stopped server")

        self._stop()
