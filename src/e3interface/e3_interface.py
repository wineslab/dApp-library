import queue
import random
import threading
import zmq
from .e3_connector import E3Connector
from .e3_logging import e3_logger
from .e3_encoder import E3Encoder

class E3Interface:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(E3Interface, cls).__new__(cls)
        return cls._instance

    def __init__(self, encoder: E3Encoder, *args, **kwargs):
        """
        Initialize E3Interface with a specific encoder.
        
        Args:
            encoder: The E3Encoder instance to use for message encoding/decoding
            **kwargs: Additional configuration parameters (link, transport, etc.)
        """
        if not hasattr(self, "initialized"):
            self.indication_callbacks = {}   # key: (dAppId, subscriptionId) -> list(callbacks)
            self.subscription_callbacks = {} # key: dAppId -> list(callbacks)
            self.xapp_control_callbacks = {} # key: dAppId -> list(callbacks)
            self.stop_event = threading.Event()
            self.initialized = True

            # Message ID management
            self._message_id_lock = threading.Lock()

            # Create an E3Connector instance based on the configuration
            self.e3_connector = E3Connector.setup_connector(kwargs.get('link', ''), kwargs.get('transport', '')) 

            e3_logger.info(f"Endpoint setup {self.e3_connector.setup_endpoint}")
            e3_logger.info(f"Endpoint inbound {self.e3_connector.inbound_endpoint}")
            e3_logger.info(f"Endpoint outbound {self.e3_connector.outbound_endpoint}")

            # Use the provided encoder directly
            self.encoder = encoder
            self.outbound_queue = queue.Queue()

    def send_setup_request(self, e3apProtocolVersion: str = "0.0.0", dAppName: str = "", dAppVersion: str = "0.0.0", vendor: str = "") -> tuple[bool, list | None]:
        """Send a setup request to the E3 interface
        
        Args:
            e3apProtocolVersion: E3AP protocol version string (e.g., "0.0.0")
            dAppName: Name of the dApp
            dAppVersion: Version of the dApp (e.g., "0.0.0")
            vendor: Vendor name (max 30 chars)
        """
        e3_logger.info(f"Send setup request for dApp '{dAppName}' version {dAppVersion} (vendor={vendor}, e3ap_version={e3apProtocolVersion})")
        msg_id = self._get_next_message_id()
        payload = self.encoder.create_setup_request(msgId=msg_id, e3apProtocolVersion=e3apProtocolVersion, dAppName=dAppName, dAppVersion=dAppVersion, vendor=vendor)
        try:
            response = self.e3_connector.send_setup_request(payload)       
        except ConnectionRefusedError as e:
            e3_logger.exception(f"Unable to connect to E3 setup endpoint, connection refused")
            return False, None

        e3_logger.info("Setup response received")
        pdu = self.encoder.decode_pdu(response)
        pdu_type, pdu_data, _msg_id = pdu
        
        if pdu_type == "setupResponse":
            e3_setup_response = pdu_data
            e3_logger.info(e3_setup_response)
            if e3_setup_response['requestId'] != msg_id:
                raise ValueError("Request id is different from the one sent!")
            outcome = e3_setup_response['responseCode'] == 'positive'
            return outcome, e3_setup_response
        else:
            e3_logger.error(f"Unexpected PDU type in the setup request {pdu_type}")
            return False, None

    def send_subscription_request(self, dappId: int, ranFunctionId: int, telemetryIds: list[int], controlIds: list[int],
                                  subscriptionTime: int | None = None, periodicity: int | None = None) -> bool:
        """Send a subscription request to the E3 interface"""
        e3_logger.info(f"Start subscription request for RAN function {ranFunctionId}")
        msg_id = self._get_next_message_id()
        proto_pdu = {
                'msgId': msg_id,
                'dappId': dappId,
                'ranFunctionId': ranFunctionId,
                'telemetryIdentifierList': telemetryIds,
                'controlIdentifierList': controlIds
            }
    
        # isinstance ensures that if fields are 0 they are still included
        if isinstance(subscriptionTime, int):
            proto_pdu['subscriptionTime'] = subscriptionTime

        if isinstance(periodicity, int):
            proto_pdu['periodicity'] = periodicity

        try:
            # Pass raw data and message ID to outbound queue for encoding
            self.outbound_queue.put(('subscription', proto_pdu))
            e3_logger.info("Subscription request queued")
            return True
        except Exception as e:
            e3_logger.error(f"Failed to send subscription request: {e}")
            return False

    def send_message_ack(self, requestId: int, responseCode: str = "positive"):
        """Send a message acknowledgment"""
        msg_id = self._get_next_message_id()

        # Pass raw data and message ID to outbound queue for encoding
        self.outbound_queue.put(('ack', {
            'msgId': msg_id,
            'requestId': requestId,
            'responseCode': responseCode
        }))
        e3_logger.debug(f"Message ACK queued for request {requestId}")
        return True
    
    def send_release_message(self, dappId: int):
        """Send a release message to end interactions with RAN"""
        msg_id = self._get_next_message_id()

        # Pass raw data and message ID to outbound queue for encoding
        self.outbound_queue.put(('release', {
            'msgId': msg_id,
            'dappId': dappId
        }))
        e3_logger.debug(f"Release Message queued for dApp {dappId}")
        return True

    def setup_connections(self):
        # Two connections: use threads so callbacks remain local callables
        self.inbound_thread = threading.Thread(target=self._inbound_connection)
        self.outbound_thread = threading.Thread(target=self._outbound_connection)
        self.inbound_thread.start()
        self.outbound_thread.start()

    def _inbound_connection(self):
        """
        Inbound is for all the messages that are coming from the RAN after the initial setup 
        """
        e3_logger.info(f'Start inbound connection')
        self.e3_connector.setup_inbound_connection()
        e3_logger.info(f'Start inbound loop')

        try:
            while not self.stop_event.is_set():
                data = self.e3_connector.receive()
                if not data:
                    e3_logger.error(f'No data received, connection closed, end')
                    break
                e3_logger.debug(f'Received data size: {len(data)}')
                # e3_logger.debug(data.hex())
                pdu = self.encoder.decode_pdu(data)
                pdu_type, pdu_data, msg_id = pdu
                e3_logger.debug(f"Data decoded")
                match pdu_type:
                    case "subscriptionResponse":
                        e3_subscription_response = pdu_data
                        e3_logger.info(
                            f"Received subscription response: {e3_subscription_response}"
                        )
                        self._handle_subscription_response(e3_subscription_response)

                    case "indicationMessage":
                        e3_indication_message = pdu_data
                        dapp_identifier = e3_indication_message['dAppIdentifier']
                        protocolData = e3_indication_message['protocolData']
                        e3_logger.debug(f"Indication message for dApp {dapp_identifier}, protocolData {len(protocolData)} bytes")
                        self._handle_indication_data(dapp_identifier, protocolData)

                    case "messageAck":
                        e3_message_ack = pdu_data
                        e3_logger.debug(f"Received message ACK: {e3_message_ack}")
                        # Just log the ACK, no correlation needed atm
                        continue

                    case "xAppControlAction":
                        e3_xapp_control_action = pdu_data
                        dapp_identifier = e3_xapp_control_action['dAppIdentifier']
                        self._handle_xapp_control_data(dapp_identifier, e3_xapp_control_action)

                    case _:
                        raise ValueError("Unrecognized PDU type ", pdu_type)
        except KeyboardInterrupt:
            e3_logger.debug("Inbound thread received SIGINT, stopping")
            self.stop_event.set()
        except zmq.error.ContextTerminated:
            e3_logger.debug("Inbound connection context terminated, exiting")
            self.stop_event.set()
        except Exception:
            e3_logger.exception(f"Error in inbound thread")
            self.stop_event.set()
        finally:
            e3_logger.info("Close inbound connection")

    def _outbound_connection(self):
        """
        Outbound is for all the messages that should go to the RAN after the initial setup 
        Messages are dApp Control Action and dApp Report Message
        """
        e3_logger.info(f'Start outbound connection')
        self.e3_connector.setup_outbound_connection()

        e3_logger.info(f'Start outbound loop')
        try:
            while not self.stop_event.is_set():
                try:
                    msg, data = self.outbound_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                e3_logger.debug(f"Outbound queue has got '{msg}', {data}")

                match msg:
                    case "control":
                        payload = self.encoder.create_control_action(
                            data["msgId"],
                            data["dappId"],
                            data["ranFunctionId"],
                            data["controlId"],
                            data["actionData"],
                        )
                    case "subscription":
                        payload = self.encoder.create_subscription_request(
                            data['msgId'], data['dappId'], data['ranFunctionId'],
                            data['telemetryIdentifierList'],
                            data['controlIdentifierList'],
                            data.get('subscriptionTime'),
                            data.get('periodicity'))
                    case "ack":
                        payload = self.encoder.create_message_ack(data['msgId'], data['requestId'], data['responseCode'])
                    case "report":
                        payload = self.encoder.create_dapp_report(data['msgId'], data['dappId'], data['ranFunctionId'], data['reportData'])
                    case "release":
                        payload = self.encoder.create_release_message(data['msgId'], data['dappId'])
                    case _:
                        raise ValueError("Unrecognized value ", msg)

                e3_logger.debug(f"Send the pdu encoded {payload}")
                self.e3_connector.send(payload)
        except KeyboardInterrupt:
            e3_logger.debug("Outbound thread received SIGINT, stopping")
            self.stop_event.set()
        except zmq.error.ContextTerminated:
            e3_logger.debug("Outbound connection context terminated, exiting")
            self.stop_event.set()
        except Exception:
            e3_logger.exception(f"Error in outbound thread")
            self.stop_event.set()
        finally:
            e3_logger.info("Close outbound connection")

    def _handle_subscription_response(self, data):
        dapp_id = data['dAppIdentifier']
        e3_logger.debug(f"DApp ID requested {dapp_id}, map status {self.subscription_callbacks}")
            
        callbacks = self.subscription_callbacks.get(dapp_id, [])
        if callbacks:
            e3_logger.debug(f"Launch {len(callbacks)} subscription callback(s) for dApp {dapp_id}")
            for callback in callbacks:
                callback(data)
        else:
            e3_logger.warning(f"No subscription callback registered for dApp {dapp_id}")

    def _handle_indication_data(self, dapp_identifier, data):
        matched_keys = [key for key in self.indication_callbacks if key[0] == dapp_identifier]
        if matched_keys:
            total_callbacks = sum(len(self.indication_callbacks[key]) for key in matched_keys)
            e3_logger.debug(f"Launch {total_callbacks} callback(s) for dApp {dapp_identifier}")
            for key in matched_keys:
                for callback in self.indication_callbacks[key]:
                    callback(dapp_identifier, data)
        else:
            e3_logger.warning(f"No indication callback registered for dApp {dapp_identifier}")

    def _handle_xapp_control_data(self, dapp_identifier, data):
        ran_function_id = data["ranFunctionIdentifier"]
        xapp_control_data = data["xAppControlData"]

        e3_logger.debug(
            f"Received xAppControlAction: "
            f"dApp={dapp_identifier}, ranFunc={ran_function_id}, "
            f"payload={len(xapp_control_data)} bytes, "
            f"xAppControlAction payload (hex): {xapp_control_data.hex()}"
        )

        callbacks = self.xapp_control_callbacks.get(dapp_identifier, [])
        if callbacks:
            e3_logger.debug(f"Launch {len(callbacks)} xApp control callback(s) for dApp {dapp_identifier}")
            for callback in callbacks:
                callback(dapp_identifier, xapp_control_data)
        else:
            e3_logger.warning(f"No xApp control callback registered for dApp {dapp_identifier}")

    def schedule_control(self, dappId: int, ranFunctionId: int, controlId: int, actionData: bytes = b""):
        msg_id = self._get_next_message_id()
        self.outbound_queue.put(('control', {
            'msgId': msg_id,
            'dappId': dappId,
            'ranFunctionId': ranFunctionId,
            'controlId': controlId,
            'actionData': actionData
        }))

    def schedule_report(self, dappId: int, ranFunctionId: int, reportData: bytes):
        msg_id = self._get_next_message_id()
        self.outbound_queue.put(('report', {
            'msgId': msg_id,
            'dappId': dappId,
            'ranFunctionId': ranFunctionId,
            'reportData': reportData
        }))

    def add_subscription_callback(self, dapp_id: int, callback):
        if dapp_id not in self.subscription_callbacks:
            e3_logger.debug(f"Add first subscription callback for dApp {dapp_id}")
            self.subscription_callbacks[dapp_id] = [callback]
        else:
            callbacks = list(self.subscription_callbacks[dapp_id])
            if callback not in callbacks:
                e3_logger.debug(f"Add additional subscription callback for dApp {dapp_id}")
                callbacks.append(callback)
                self.subscription_callbacks[dapp_id] = callbacks
            else:
                e3_logger.warning(f"Subscription callback already registered for dApp {dapp_id}, skipping")


    def remove_subscription_callback(self, dapp_id: int, callback=None):
        if dapp_id in self.subscription_callbacks:
            if callback is None:
                e3_logger.debug(f"Remove all subscription callbacks for dApp {dapp_id}")
                del self.subscription_callbacks[dapp_id]
            else:
                callbacks = list(self.subscription_callbacks[dapp_id])
                if callback in callbacks:
                    e3_logger.debug(f"Remove specific subscription callback for dApp {dapp_id}")
                    callbacks.remove(callback)
                    if callbacks:
                        self.subscription_callbacks[dapp_id] = callbacks
                    else:
                        del self.subscription_callbacks[dapp_id]
                else:
                    e3_logger.warning(f"Specific subscription callback not found for dApp {dapp_id}")
        else:
            e3_logger.warning(f"No subscription callbacks found for dApp {dapp_id}")

    def add_indication_callback(self, dapp_id: int, subscription_id: int, callback):
        key = (dapp_id, subscription_id)

        if key not in self.indication_callbacks:
            e3_logger.debug(f"Add first indication callback for dApp {dapp_id}, subscription {subscription_id}")
            self.indication_callbacks[key] = [callback]
        else:
            callbacks = list(self.indication_callbacks[key])
            if callback not in callbacks:
                e3_logger.debug(f"Add additional indication callback for dApp {dapp_id}, subscription {subscription_id}")
                callbacks.append(callback)
                self.indication_callbacks[key] = callbacks
            else:
                e3_logger.warning(
                    f"Indication callback already registered for dApp {dapp_id}, subscription {subscription_id}, skipping"
                )

    def remove_indication_callback(self, dapp_id: int, subscription_id: int | None = None, callback=None):
        if subscription_id is None:
            # Remove all entries for this dapp_id
            keys_to_remove = [key for key in self.indication_callbacks if key[0] == dapp_id]
            if keys_to_remove:
                e3_logger.debug(f"Remove all indication callbacks for dApp {dapp_id}")
                for key in keys_to_remove:
                    del self.indication_callbacks[key]
            else:
                e3_logger.warning(f"No indication callbacks found for dApp {dapp_id}")

        elif callback is not None:
            # Remove specific callback from any key matching dapp_id
            found = False
            keys_to_check = [key for key in self.indication_callbacks if key[0] == dapp_id]
            for key in keys_to_check:
                callbacks = list(self.indication_callbacks[key])
                if callback in callbacks:
                    e3_logger.debug(f"Remove specific callback for dApp {dapp_id}, subscription {key[1]}")
                    callbacks.remove(callback)
                    if callbacks:
                        self.indication_callbacks[key] = callbacks
                    else:
                        del self.indication_callbacks[key]
                    found = True
                    break
            if not found:
                e3_logger.warning(f"Specific callback not found for dApp {dapp_id}")

        else:
            # subscription_id is present, callback is None: remove the specific key
            key = (dapp_id, subscription_id)
            if key in self.indication_callbacks:
                e3_logger.debug(f"Remove all callbacks for dApp {dapp_id}, subscription {subscription_id}")
                del self.indication_callbacks[key]
            else:
                e3_logger.warning(f"No indication callbacks found for dApp {dapp_id}, subscription {subscription_id}")

    def add_xapp_control_callback(self, dapp_id: int, subscription_id: int, callback):
        if dapp_id not in self.xapp_control_callbacks:
            e3_logger.debug(f"Add first xApp control callback for dApp {dapp_id}")
            self.xapp_control_callbacks[dapp_id] = [callback]
        else:
            callbacks = list(self.xapp_control_callbacks[dapp_id])
            if callback not in callbacks:
                e3_logger.debug(f"Add additional xApp control callback for dApp {dapp_id}")
                callbacks.append(callback)
                self.xapp_control_callbacks[dapp_id] = callbacks
            else:
                e3_logger.warning(f"xApp control callback already registered for dApp {dapp_id}, skipping")

    def remove_xapp_control_callback(self, dapp_id: int, subscription_id: int | None = None,  callback=None):
        if dapp_id in self.xapp_control_callbacks:
            if callback is None:
                e3_logger.debug(f"Remove all xApp control callbacks for dApp {dapp_id}")
                del self.xapp_control_callbacks[dapp_id]
            else:
                callbacks = list(self.xapp_control_callbacks[dapp_id])
                if callback in callbacks:
                    e3_logger.debug(f"Remove specific xApp control callback for dApp {dapp_id}")
                    callbacks.remove(callback)
                    if callbacks:
                        self.xapp_control_callbacks[dapp_id] = callbacks
                    else:
                        del self.xapp_control_callbacks[dapp_id]
                else:
                    e3_logger.warning(f"Specific xApp control callback not found for dApp {dapp_id}")
        else:
            e3_logger.warning(f"No xApp control callbacks found for dApp {dapp_id}")


    def _get_next_message_id(self):
        """Generate next message ID in thread-safe manner (1..1000)"""
        with self._message_id_lock:
            return random.randint(1, 1000)

    def terminate_connections(self):
        e3_logger.info("Stop event")
        self.stop_event.set()

        # Dispose connector early to unblock any blocking recv/send calls
        try:
            self.e3_connector.dispose()
        except Exception:
            e3_logger.debug("Error disposing connector during shutdown")

        if hasattr(self, "inbound_thread") and self.inbound_thread.is_alive():
            self.inbound_thread.join(timeout=2)
            if self.inbound_thread.is_alive():
                e3_logger.warning("Inbound thread did not terminate gracefully")

        if hasattr(self, "outbound_thread") and self.outbound_thread.is_alive():
            self.outbound_thread.join(timeout=2)
            if self.outbound_thread.is_alive():
                e3_logger.warning("Outbound thread did not terminate gracefully")

    def __del__(self):
        if not self.stop_event.is_set():
            self.terminate_connections()
