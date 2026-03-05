#!/usr/bin/env python3
"""
dApp Base class that should be extended.
This class implements the E3AP and is a wrapper for the access to the E3Interface class.+
"""

__author__ = "Andrea Lacava"

from abc import ABC, abstractmethod
import threading
import time
from e3interface.e3_interface import E3Interface
from e3interface.e3_encoder import AsnE3Encoder, JsonE3Encoder
from e3interface.e3_logging import dapp_logger

class DApp(ABC):
    e3_interface: E3Interface

    # TODO at the moment this implementation supports only one RAN Function ID, this will be extended to cover the multiple cases
    RAN_FUNCTION_ID = 0
    TELEMETRY_ID = []
    CONTROL_ID = []
    
    def __init__(self, dapp_name: str = "Unknown dApp", dapp_version: str = "0.0.0", 
                 vendor: str = "Unknown", e3ap_protocol_version: str = "1.0.0",
                 link: str = 'posix', transport: str = 'ipc', callbacks: list = [], 
                 encoding_method: str = 'asn1', **kwargs):
        super().__init__()
        self.dapp_name = dapp_name
        self.dapp_version = dapp_version
        self.vendor = vendor
        self.e3ap_protocol_version = e3ap_protocol_version
        self.encoding_method = encoding_method        
        match self.encoding_method:
            case "asn1":
                encoder = AsnE3Encoder()
            case "json":
                encoder = JsonE3Encoder()
            case _:
                raise ValueError(f"Unsupported encoding method: {self.encoding_method}")

        self.e3_interface = E3Interface(encoder=encoder, link=link, transport=transport)
        self.stop_event = threading.Event()

        dapp_logger.info(f'Using {link} and {transport}')

        self.callbacks = callbacks

    def setup_connection(self):
        attempts = 3
        
        while attempts > 0:
            outcome, response = self.e3_interface.send_setup_request(
                e3apProtocolVersion=self.e3ap_protocol_version,
                dAppName=self.dapp_name,
                dAppVersion=self.dapp_version,
                vendor=self.vendor
            )
            dapp_logger.info(f'E3 Setup Response outcome: {outcome}')
            
            if outcome:
                dapp_logger.debug(f'E3 Setup Response: {response}')
                self.dapp_id = response['dAppIdentifier']
                self.e3_interface.setup_connections()
                return outcome, response
            
            attempts-=1       
            dapp_logger.warning('RAN refused setup or dApp was not able to connect, waiting 1s')
            time.sleep(1)
        
        dapp_logger.error('dApp unable to connect to RAN')
        return outcome, None

    @classmethod
    def check_sm_ids(cls, ranFunctionId: int = 0, telemetryIds: list[int] = [], controlIds: list[int] = []) -> bool:
        # TODO this function might not make sense as written in this way, it needs to be clarified why this check is needed and what to do if it is failed
        return (
            ranFunctionId == cls.RAN_FUNCTION_ID
            and all(telemetry_id in telemetryIds for telemetry_id in cls.TELEMETRY_ID)
            and all(control_id in controlIds for control_id in cls.CONTROL_ID)
        )

    @abstractmethod
    def _decode_ran_function_data(self, data_bytes: bytes) -> dict | None:
        pass

    def decode_ran_function_data(self, ran_function_data) -> dict | None:
        """Decode the opaque `ranFunctionData` attached to a SetupResponse entry.

        The payload may be:
        - ASN.1 PER encoded (bytes) when using ASN.1
        - JSON-encoded string

        This method calls internally _decode_ran_function_data() that should be overridden by children classes

        Returns the decoded dict or None on failure.
        """
        if ran_function_data is None:
            return None

        # Normalize to bytes
        data_bytes = None
        if isinstance(ran_function_data, str):
            data_bytes = ran_function_data.encode("utf-8")
        elif isinstance(ran_function_data, (bytes, bytearray)):
            data_bytes = bytes(ran_function_data)
        else:
            return None

        try:
            return self._decode_ran_function_data(data_bytes)
        except Exception:
            dapp_logger.exception("Failed to decode ranFunctionData payload")
            return None

    def manage_subscription_response(self, data):
        """Called when a subscription response arrives."""
        dapp_logger.debug(f"Subscription response received: {data}")
        response_code = data['responseCode']
        
        # TODO this logic should become more smart to handle contract and subscriptions more granuarly
        if response_code == 'positive':
            dapp_logger.info("Positive subscription response, registering indication callback.")
            subscription_id = data['subscriptionId']
            self.e3_interface.add_indication_callback(self.dapp_id, subscription_id, self._handle_indication)
            self.e3_interface.add_xapp_control_callback(self.dapp_id, subscription_id, self._handle_xapp_control)
        else:
            dapp_logger.warning(f"Subscription response not positive: {data}")

    def send_subscription_request(self, subscriptionTime: int | None = None, periodicity: int | None = None) -> bool: 
        self.e3_interface.add_subscription_callback(self.dapp_id, self.manage_subscription_response)
        dapp_logger.debug(f"Subscription callbacks: {self.e3_interface.subscription_callbacks}")
        scheduled = self.e3_interface.send_subscription_request(
                self.dapp_id,
                self.RAN_FUNCTION_ID,
                self.TELEMETRY_ID,
                self.CONTROL_ID,
                subscriptionTime,
                periodicity)
        
        dapp_logger.debug(f"Subscription request for {self.RAN_FUNCTION_ID}{'' if scheduled else ' not'} scheduled")

        return scheduled

    @abstractmethod
    def _handle_xapp_control(self, dapp_identifier: int, data: bytes):
        # This in the future might become a class
        pass

    @abstractmethod
    def _handle_indication(self, dapp_identifier: int, data: bytes):
        # This in the future might become a class
        pass

    @abstractmethod
    def _control_loop(self):
        pass

    def control_loop(self):
        dapp_logger.debug(f"Start control loop")
        try:
            while not self.stop_event.is_set():
                self._control_loop()
        except KeyboardInterrupt:
            dapp_logger.info("Keyboard interrupt received, closing dApp")
        finally:
            self.stop()

    @abstractmethod
    def _stop(self):
        pass

    def stop(self):
        try:
            dapp_logger.info('Sending Message release to unregister dApp from RAN')
            self.e3_interface.send_release_message(self.dapp_id)
            time.sleep(1.5)
            dapp_logger.info('Stopping dApp')
        except Exception as e:
            dapp_logger.error(f'Failed during dApp stop: {e}')
        finally:
            self.stop_event.set()
            time.sleep(1)
            self.e3_interface.terminate_connections()
            dapp_logger.info("Stop. Program exit")
            self._stop()
