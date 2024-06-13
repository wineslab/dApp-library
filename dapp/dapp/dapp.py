#!/usr/bin/env python3
"""
Module Docstring
"""

__author__ = "Andrea Lacava"
__version__ = "0.1.0"
__license__ = "MIT"

from abc import ABC, abstractmethod


class DApp(ABC):
    
    def __init__(self) -> None:
        super().__init__()


    @abstractmethod
    def method_to_override(self):
        # should be overridden
        pass
    

    def main_method(self):
        # Call the overridden method
        self.method_to_override()
        
        # Additional logic for the main method
        print("Main method logic")

    # create destructor
    # we should join all the threads created from the dApp here
    def __del__(self):
        print('ciao')