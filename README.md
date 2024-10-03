# spear-dApp

password of the colosseum container of dApp is `spear`

## Usage

### Building OAI
```
./build_oai -w USRP --ninja --gNB --build-e3
```

### Configuration for OAI
- Make sure that do_SRS flag is set to 0

Currently tested with the config file `./gNB.conf`

### dApp kickoff

OAI should start _before_ running the dApp

```python 
python3 src/dapp/dapp.py
```

There are two possible arguments:
- `ota` bool (false): If true, use the FFT and noise floor value for OTA else use the ones of Colosseum
- `control` bool (false): If set to true, performs the PRB masking
- `gui` bool (false): If set to true, creates and show the sensed spectrum