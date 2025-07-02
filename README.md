# dApp Library

A complete tutorial on how to deploy a dApp can be found on the [OpenRAN Gym website](https://openrangym.com/tutorials/dapps-oai). Please refer to that guide to instrument your system.

## Installation

### Python package installation (recommended)

```
pip3 install dapps[all]
```

The dApp can be found in the `examples` directory or it can be downloaded from this repository.

### Manual installation (use only for library development)

Clone the repository and install it manually with:

```
hatch build
pip3 install dist/*.tar.gz
```

### Launch the Spectrum Sharing dApp example

OAI should start _before_ running the dApp

```python 
python3 examples/spectrum_dapp.py
```

This dApp implements a spectrum sharing use case discussed in [our paper](https://doi.org/10.1016/j.comnet.2025.111342) ([here for the archive version](https://arxiv.org/pdf/2501.16502)).
The dApp parameters can be controlled through the following command-line arguments:

- `--link` (str, default: 'zmq', choices: [layer.value for layer in E3LinkLayer]): Specify the link layer to be used  
- `--transport` (str, default: 'ipc', choices: [layer.value for layer in E3TransportLayer]): Specify the transport layer to be used  
- `--save-iqs` (store_true, default: False): Specify if this is data collection run or not. In the first case I/Q samples will be saved  
- `--control` (store_true, default: False): Set whether to perform control of PRB  
- `--noise-floor-threshold` (int, default: None): Set the noise floor threshold for determining the presence of incumbents and for detecting the PRBs affected.
- `--ota` (store_true, default: False): Specify if the setup used is OTA or on Colosseum for determining the noise floor threshold. If the `noise_floor_threshold` parameter is specified, this parameter is ignored
- `--energy-gui` (store_true, default: False): Set whether to enable the energy GUI  
- `--iq-plotter-gui` (store_true, default: False): Set whether to enable the IQ Plotter GUI  
- `--demo-gui` (store_true, default: False): Set whether to enable the Demo GUI  
- `--num-prbs` (int, default: 106): Number of PRBs  
- `--num-subcarrier-spacing` (int, default: 30): Subcarrier spacing in kHz (FR1 is 30)  
- `--e` (store_true, default: False): Set if 3/4 sampling for FFT size is set on the gNB (-E option on OAI)  
- `--center-freq` (float, default: 3.6192e9): Center frequency in Hz  
- `--timed` (store_true, default: False): Run with a 5-minute time limit  
- `--model` (str, default: ''): Path to the CNN model file to be used  
- `--time-window` (int, default: 5): Number of input vectors to pass to the CNN model  
- `--moving-avg-window` (int, default: 30): Window size (in samples) for the moving average used to detect energy peaks in the spectrum  
- `--extraction-window` (int, default: 600): Number of samples to retain after detecting an energy peak


If you use the dApp concept and/or the framework to develop your own dApps, please cite the following paper:

```text
@article{LACAVA2025111342,
title = {{dApps: Enabling Real-Time AI-based Open RAN Control}},
journal = {Computer Networks},
pages = {111342},
year = {2025},
issn = {1389-1286},
doi = {https://doi.org/10.1016/j.comnet.2025.111342},
url = {https://www.sciencedirect.com/science/article/pii/S1389128625003093},
author = {Andrea Lacava and Leonardo Bonati and Niloofar Mohamadi and Rajeev Gangula and Florian Kaltenberger and Pedram Johari and Salvatore D’Oro and Francesca Cuomo and Michele Polese and Tommaso Melodia},
keywords = {Open RAN, dApps, Real-time control loops, Radio Resource Management (RRM), Spectrum sharing, Positioning, Integrated Sensing and Communication (ISAC)},
abstract = {Open Radio Access Networks (RANs) leverage disaggregated and programmable RAN functions and open interfaces to enable closed-loop, data-driven radio resource management. This is performed through custom intelligent applications on the RAN Intelligent Controllers (RICs), optimizing RAN policy scheduling, network slicing, user session management, and medium access control, among others. In this context, we have proposed dApps as a key extension of the O-RAN architecture into the real-time and user-plane domains. Deployed directly on RAN nodes, dApps access data otherwise unavailable to RICs due to privacy or timing constraints, enabling the execution of control actions within shorter time intervals. In this paper, we propose for the first time a reference architecture for dApps, defining their life cycle from deployment by the Service Management and Orchestration (SMO) to real-time control loop interactions with the RAN nodes where they are hosted. We introduce a new dApp interface, E3, along with an Application Protocol (AP) that supports structured message exchanges and extensible communication for various service models. By bridging E3 with the existing O-RAN E2 interface, we enable dApps, xApps, and rApps to coexist and coordinate. These applications can then collaborate on complex use cases and employ hierarchical control to resolve shared resource conflicts. Finally, we present and open-source a dApp framework based on OpenAirInterface (OAI). We benchmark its performance in two real-time control use cases, i.e., spectrum sharing and positioning in a 5th generation (5G) Next Generation Node Base (gNB) scenario. Our experimental results show that standardized real-time control loops via dApps are feasible, achieving average control latency below 450microseconds and allowing optimal use of shared spectral resources.}
}
```
