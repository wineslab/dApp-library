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
pip3 install "dist/*.tar.gz[all]"
```

### Launch the Spectrum Sharing dApp example

OAI should start _before_ running the dApp

```python 
python3 examples/spectrum_dapp.py
```

This dApp implements a spectrum sharing use case discussed in [our paper](https://doi.org/10.1016/j.comnet.2025.111342) ([here for the archive version](https://arxiv.org/pdf/2501.16502)).

**Command-line arguments:**

**Connection Configuration:**
- `--link` - Specify the link layer to be used. Options: `zmq`, `posix` (default: `zmq`)
- `--transport` - Specify the transport layer to be used. Options: `ipc`, `tcp` (default: `ipc`)

**Data Collection and Control:**
- `--save-iqs` - Enable data collection mode. When set, I/Q samples will be saved to disk for later analysis
- `--control` - Enable PRB control. When set, the dApp will actively control spectrum allocation based on detected interference

**Spectrum Sensing Configuration:**
- `--noise-floor-threshold` - Set the noise floor threshold (in dB) for determining the presence of incumbents and for detecting the PRBs affected. If not specified, will be auto-detected based on environment
- `--ota` - Specify if the setup is Over-The-Air (OTA) or on Colosseum testbed for using a precalculated noise floor threshold. If `--noise-floor-threshold` is specified, this parameter is ignored

**Visualization Options:**
- `--energy-gui` - Enable the energy spectrum visualization GUI
- `--iq-plotter-gui` - Enable the I/Q constellation plotter GUI
- `--demo-gui` - Enable the demonstration GUI showing real-time spectrum occupancy

**Radio Configuration:**
- `--num-prbs` - Number of Physical Resource Blocks in the channel (default: 106)
- `--num-subcarrier-spacing` - Subcarrier spacing in kHz. Use 30 for FR1 (sub-6 GHz) (default: 30)
- `--e` - Enable 3/4 FFT sampling mode. Set this flag if the gNB was started with the `-E` option
- `--center-freq` - Center frequency in Hz (default: 3.6192e9, which is 3.6192 GHz)

**Execution Control:**
- `--timed SECONDS` - Run the dApp with a time limit (in seconds). Set to 0 for no limit (default: 0)

**AI/ML Model Configuration:**
- `--model` - Path to a pre-trained CNN model file for spectrum classification. If not provided, uses rule-based detection
- `--time-window` - Number of consecutive spectrum vectors to use as input for the CNN model (default: 5)

**Signal Processing Parameters:**
- `--moving-avg-window` - Window size (in samples) for the moving average filter used to smooth energy measurements and detect peaks in the spectrum (default: 30)
- `--extraction-window` - Number of samples to retain after detecting an energy peak for further analysis (default: 600)
- `--sampling-threshold` - Down-sampling ratio applied to IQ sensing. An IQ vector will be delivered every Nth sensing (default: 5)

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

## E3 Service Model Generator

The `tools/generate_sm.sh` script helps you create skeleton code for new E3 Service Models. This automates the creation of boilerplate code for both the agent (OAI) and dApp sides.

### Usage

**Interactive mode (recommended for first-time users):**
```bash
cd tools
./generate_sm.sh --interactive
```

**Command-line mode:**
```bash
./generate_sm.sh -n <sm_name> -a <asn_file> [OPTIONS]
```

**Options:**
- `-n, --name NAME` - Service Model name (required)
- `-a, --asn-file FILE` - ASN.1 definition file path (required)
- `-t, --target TARGET` - Target: `agent`, `dapp`, or `both` (default: both)
- `-r, --ran-functions IDS` - Comma-separated RAN function IDs (e.g., 2,3,4)
- `-f, --format FORMAT` - Encoding format: `asn1` or `json` (default: asn1)
- `--agent-root PATH` - Path to OAI repository (or set `AGENT_ROOT` env var)
- `-i, --interactive` - Interactive mode with prompts
- `-h, --help` - Show help message

**Examples:**
```bash
# Generate both agent and dApp code for positioning SM
cd tools
./generate_sm.sh -n positioning -a positioning.asn -r 2 --agent-root /path/to/spear-openairinterface5g

# Generate only dApp code for KPM SM
./generate_sm.sh -n kpm -a kpm.asn -r 3,4 -t dapp

# Interactive mode
./generate_sm.sh --interactive
```

**Generated Structure:**

For agent-side (when `--target agent` or `both`):
```
spear-openairinterface5g/openair2/E3AP/service_models/<sm_name>_sm/
├── <sm_name>_sm.h          # Main SM header
├── <sm_name>_sm.c          # Main SM implementation
├── <sm_name>_enc.h/.c      # Encoding functions
├── <sm_name>_dec.h/.c      # Decoding functions
├── CMakeLists.txt          # Build configuration
└── MESSAGES/
    ├── CMakeLists.txt
    └── ASN1/V1/
        ├── e3sm_<sm_name>.asn
        └── e3sm_<sm_name>.cmake
```

For dApp-side (when `--target dapp` or `both`):
```
spear-dApp/src/<sm_name>/
├── __init__.py
├── <sm_name>_dapp.py       # Main dApp implementation
└── defs/
    └── e3sm_<sm_name>.asn  # ASN.1 schema
```

### Contributing

Contributions are welcome, but they must follow the guidelines in [CONTRIBUTING.md](./CONTRIBUTING.md)