# spear-dApp

password of the colosseum container of dApp is `spear`

## Usage

### Building OAI
```
./build_oai -w USRP --ninja --gNB --build-e3
```

### System configuration

If we need the dApp to run in a different user than OAI (e.g., $USER vs root) we need to create a specific unix users group called `dapp` and we assign root and user to this group to enable shared UDS through a dedicated foldert:
```
sudo groupadd dapp
sudo usermod -aG dapp root
sudo usermod -aG dapp $USER

# check the groups
groups root $USER
```

Folder creation (path is not important, but it should be the same in OAI and the dApp)

```
mkdir -p /tmp/dapps
sudo chown :dapp /tmp/dapps
sudo chmod g+ws /tmp/dapps
```


### Configuration for OAI
- Make sure that do_SRS flag is set to 0

We support two different configuration for OTA and colosseum.

### dApp kickoff

OAI should start _before_ running the dApp

```python 
python3 src/dapp/dapp.py
```

There are two possible arguments:
- `ota` bool (false): If true, use the OAI and spectrum configurations for OTA else use the ones of Colosseum
- `control` bool (false): If set to true, performs the PRB masking
- `energy-gui` bool (false): If set to true, creates and show the energy spectrum
- `iq-plotter-gui`bool (false): If set to true, creates and show the sensed spectrum