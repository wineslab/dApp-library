#!/usr/bin/env python3
"""
dApp for NVIDIA Aerial
"""

__author__ = "Andrea Lacava"

import time
import os
import json
import asn1tools
import jsonschema
# np.set_printoptions(threshold=sys.maxsize)

from typing import override
from dapp.dapp import DApp
from e3interface.e3_logging import dapp_logger

class NvidiaDApp(DApp):

    # IDs of interest for this dApp (TBD)
    RAN_FUNCTION_ID = 0
    TELEMETRY_ID = []
    CONTROL_ID = []

    def __init__(self, dapp_name: str = "Nvidia", dapp_version: str = "1.0.0",
                 vendor: str = None, e3ap_protocol_version: str = "1.0.0",
                 link: str = 'zmq', transport: str = 'tcp', encoding_method: str = "json", **kwargs):
        super().__init__(dapp_name=dapp_name, dapp_version=dapp_version, vendor=vendor,
                         e3ap_protocol_version=e3ap_protocol_version, link=link, transport=transport, 
                         encoding_method=encoding_method, **kwargs) 


        # Initialize spectrum encoder based on encoding method
        self._init_sm_encoder()

        # Check dApp configuration

    def _init_sm_encoder(self):
        """Initialize the spectrum encoder based on the encoding method"""
        match self.encoding_method:
            case "asn1":
                asn_file_path = os.path.join(os.path.dirname(__file__), "defs", "e3sm_nvidia.asn")
                self.sm_encoder = asn1tools.compile_files(asn_file_path, codec="per")
                dapp_logger.info("ASN encoder initialized")
            case "json":
                json_schema_path = os.path.join(os.path.dirname(__file__), "defs", "e3sm_nvidia.json")
                with open(json_schema_path, 'r') as f:
                    self.nvidia_schema = json.load(f)
                self.nvidia_resolver = jsonschema.RefResolver.from_schema(self.nvidia_schema)
                self.sm_encoder = "json"  # Marker to indicate JSON mode
                dapp_logger.info("JSON encoder initialized")
            case _:
                raise ValueError(f"Unsupported encoding method: {self.encoding_method}")

    def get_data_from_ran(self, dapp_identifier, data):
        dapp_logger.debug(f'Triggered callback for dApp {dapp_identifier}')
        indication_message = data
        
        dapp_logger.debug("Received indication message")
        dapp_logger.debug(indication_message)

    @override
    def _control_loop(self):
        # Just sleep to avoid busy-waiting
        try:
           time.sleep(1)        
        except Exception:
            dapp_logger.exception("[NVIDIA] Error in the control loop")

    @override
    def _stop(self):        
        # Nothing to do here atm
        pass
