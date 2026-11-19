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
from e3interface.libe3_agent import CAUSE_UNSUPPORTED_IDENTIFIER
from e3interface.e3_logging import dapp_logger

class DApp(ABC):
    e3_interface: E3Interface

    # TODO at the moment this implementation supports only one RAN Function ID, this will be extended to cover the multiple cases
    RAN_FUNCTION_ID = 0
    TELEMETRY_ID = []
    CONTROL_ID = []

    # RAN function id -> the control ids it advertised in the setup response. Set by
    # setup_connection(); empty until then.
    ran_function_controls: dict[int, set[int]] = {}
    
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
        if encoding_method not in ("asn1", "json", "protobuf"):
            raise ValueError(f"Unsupported encoding method: {encoding_method}")

        # libe3 owns E3AP (transport + wire encoding); the encoding string is
        # passed straight through and must match the gNB E3Configuration.
        self.e3_interface = E3Interface(
            link=link, transport=transport, encoding=encoding_method)
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
                self.ran_function_controls = {
                    rf['ranFunctionIdentifier']: set(rf.get('controlIdentifierList') or [])
                    for rf in response.get('ranFunctionList') or []
                }
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
            periodicity = data.get('periodicity')
            if periodicity is None:
                dapp_logger.info("Subscription periodicity not reported by the RAN")
            else:
                dapp_logger.info(
                    f"The RAN keeps at least {periodicity} us between indications")
            subscription_id = data['subscriptionId']
            self.e3_interface.add_indication_callback(self.dapp_id, subscription_id, self._handle_indication)
            self.e3_interface.add_xapp_control_callback(self.dapp_id, subscription_id, self._handle_xapp_control)
        else:
            dapp_logger.warning(f"Subscription response not positive: {data}")

    def retry_without_refused_controls(self, ran_function_id: int, telemetry_ids: list[int],
                                       control_ids: list[int], subscription_time: int | None = None,
                                       periodicity: int | None = None, timeout: float = 5.0):
        """After the RAN refused a subscription: drop the controls it does not offer, ask again.

        A libe3 RAN refuses a subscription that names a control id its RAN function does not
        advertise (cause ``unsupportedIdentifier``). The control is then disabled for this dApp,
        so a later :meth:`E3Interface.schedule_control` for it is skipped instead of sent, and the
        subscription is repeated with the controls that remain.

        Returns the verdict of the new subscription (True accepted, False refused, None no
        response in ``timeout``), or False without asking again when the refusal is not about
        controls: another cause, a telemetry id, or nothing left to subscribe to.
        """
        refusal = self.e3_interface.subscription_refusal(ran_function_id) or {}
        cause = refusal.get('cause')
        advertised = self.ran_function_controls.get(ran_function_id)
        refused = [c for c in control_ids if advertised is not None and c not in advertised]
        # A cause of None is a RAN that gave none (Aerial) or an older libe3py: the
        # advertised ids still tell which controls cannot be the reason.
        if not refused or cause not in (None, CAUSE_UNSUPPORTED_IDENTIFIER):
            return False
        for control_id in refused:
            self.e3_interface.disable_control(ran_function_id, control_id)
        dapp_logger.warning(
            f"RAN function {ran_function_id} does not offer control(s) {refused}"
            f"{': ' + refusal['detail'] if refusal.get('detail') else ''}; "
            f"disabled, subscribing again without them")
        remaining = [c for c in control_ids if c not in refused]
        if not remaining and not telemetry_ids:
            return False
        if not self.e3_interface.send_subscription_request(
                self.dapp_id, ran_function_id, telemetry_ids, remaining,
                subscription_time, periodicity):
            return False
        return self.e3_interface.wait_for_subscription_result(ran_function_id, timeout=timeout)

    def send_subscription_request(self, subscriptionTime: int | None = None, periodicity: int | None = None) -> bool:
        self.e3_interface.add_subscription_callback(self.dapp_id, self.manage_subscription_response)

        self.e3_interface.add_indication_callback(self.dapp_id, 0, self._handle_indication)
        self.e3_interface.add_xapp_control_callback(self.dapp_id, 0, self._handle_xapp_control)
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
    def _handle_xapp_control(self, dapp_identifier: int, data: bytes, sequence_id: int = 0):
        """Handle a control addressed to this dApp by an xApp.

        ``sequence_id`` is the correlation id of the xApp procedure. Echo it on
        the control re-issued to the RAN so the RAN can report back when the
        decision reached the air. Implementations written before the id existed
        may still take two arguments; E3Interface detects that and calls them
        the old way.
        """
        # This in the future might become a class
        pass

    @abstractmethod
    def _handle_indication(self, dapp_identifier: int, ran_function_id: int, data: bytes):
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
