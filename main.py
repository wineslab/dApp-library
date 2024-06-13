#!/usr/bin/env python3
"""
Module Docstring
"""

__author__ = "Andrea Lacava"
__version__ = "0.1.0"
__license__ = ""

import argparse

from spear.spear import SpearApp


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

    spearDApp.run()
