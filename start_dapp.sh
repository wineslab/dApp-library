#!/bin/bash
python3 examples/spectrum_dapp.py --demo-gui --center-freq 3755e6 --use-adaptive-noise-floor --noise-floor-threshold 25 --embargo-timeout-secs 10 --transport tcp --toggle-period 2000000 --control
