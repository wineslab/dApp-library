import os
import multiprocessing
import queue
import threading
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
            self.callbacks = {}  # Dictionary where keys are dApp IDs and values are lists of callbacks
            self.stop_event = multiprocessing.Event()
            self.initialized = True
            
            # Message ID management
            self._message_id_counter = 1
            self._message_id_lock = threading.Lock()

            # Create an E3Connector instance based on the configuration
            self.e3_connector = E3Connector.setup_connector(kwargs.get('link', ''), kwargs.get('transport', '')) 
            
            e3_logger.info(f"Endpoint setup {self.e3_connector.setup_endpoint}")
            e3_logger.info(f"Endpoint inbound {self.e3_connector.inbound_endpoint}")
            e3_logger.info(f"Endpoint outbound {self.e3_connector.outbound_endpoint}")

            # Use the provided encoder directly
            self.encoder = encoder
            self.outbound_queue = multiprocessing.Queue()
    
    def send_setup_request(self, dappId: int = 1) -> bool | list:
        """Send a setup request to the E3 interface"""
        e3_logger.info("Start setup request")
        msg_id = self._get_next_message_id()
        payload = self.encoder.create_setup_request(dappId, msgId=msg_id)
        try:
            response = self.e3_connector.send_setup_request(payload)
        except ConnectionRefusedError as e:
            e3_logger.error(f"Unable to connect to E3 setup endpoint, connection refused: {e}")
            return False, None

        e3_logger.info("Setup response received")
        pdu = self.encoder.decode_pdu(response)
        if pdu[0] == "setupResponse":
            e3_setup_response = pdu[1]
            e3_logger.info(e3_setup_response)
            outcome = e3_setup_response['responseCode'] == 'positive'
            return outcome, e3_setup_response.get('ranFunctionList', None)
        else:
            e3_logger.error(f"Unexpected PDU type {pdu[0]}")
            return False, None

    def send_subscription_request(self, ranFunctionId: int, dappId: int = 1, actionType: str = "insert") -> bool:
        """Send a subscription request to the E3 interface"""
        e3_logger.info(f"Start subscription request for RAN function {ranFunctionId}")
        msg_id = self._get_next_message_id()
        
        try:
            # Pass raw data and message ID to outbound queue for encoding
            self.outbound_queue.put(('subscription', {
                'msgId': msg_id,
                'dappId': dappId,
                'actionType': actionType,
                'ranFunctionId': ranFunctionId
            }))
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

    def setup_connections(self):
        # Two connections, one for the inbound the other for the outbound
        self.inbound_process =  multiprocessing.Process(target=self._inbound_connection)
        self.outbound_process = multiprocessing.Process(target=self._outbound_connection)
        
        self.inbound_process.start()
        self.outbound_process.start()    

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
                e3_logger.debug(f"Data decoded")
                match pdu[0]:
                    case "indicationMessage":
                        e3_indication_message = pdu[1]
                        dapp_identifier = e3_indication_message['dAppIdentifier']
                        protocolData = e3_indication_message['protocolData']
                        e3_logger.debug(f"Indication message for dApp {dapp_identifier}, protocolData {len(protocolData)} bytes")
                        self._handle_incoming_data(dapp_identifier, protocolData)

                    case "subscriptionResponse":
                        e3_subscription_response = pdu[1]
                        e3_logger.info(f"Received subscription response: {e3_subscription_response}")
                        # Just log the response, no correlation needed

                    case "messageAck":
                        e3_message_ack = pdu[1]
                        e3_logger.debug(f"Received message ACK: {e3_message_ack}")
                        # Just log the ACK, no correlation needed
                    
                    case "xAppControlAction":
                        e3_xapp_control_action = pdu[1]
                        # not in the ASN yet but already theorized for E2SM
                        raise NotImplementedError()

                    case _:
                        raise ValueError("Unrecognized PDU type ", pdu[0])
                    
        except Exception as e:
            e3_logger.error(f"Error in inbound thread: {e}")
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
                        payload = self.encoder.create_control_action(data['msgId'], data['dappId'], data['ranFunctionId'], data['actionData'])
                    case "subscription":
                        payload = self.encoder.create_subscription_request(data['msgId'], data['dappId'], data['actionType'], data['ranFunctionId'])
                    case "ack":
                        payload = self.encoder.create_message_ack(data['msgId'], data['requestId'], data['responseCode'])
                    case "report":
                        payload = self.encoder.create_dapp_report(data['msgId'], data['reportData'])
                    case _:
                        raise ValueError("Unrecognized value ", msg)

                e3_logger.debug(f"Send the pdu encoded {payload}")
                self.e3_connector.send(payload)

        except Exception as e:
            e3_logger.error(f"Error outbound thread: {e}")
            self.stop_event.set()
        finally:
            e3_logger.info("Close outbound connection")

    def _handle_incoming_data(self, dapp_identifier, data):
        if dapp_identifier in self.callbacks:
            e3_logger.debug(f"Launch {len(self.callbacks[dapp_identifier])} callback(s) for dApp {dapp_identifier}")
            for callback in self.callbacks[dapp_identifier]:
                callback(dapp_identifier, data)
        else:
            e3_logger.warning(f"No callback registered for dApp {dapp_identifier}")
                
    def schedule_control(self, dappId: int, ranFunctionId: int, actionData: bytes):
        msg_id = self._get_next_message_id()
        self.outbound_queue.put(('control', {
            'msgId': msg_id,
            'dappId': dappId,
            'ranFunctionId': ranFunctionId,
            'actionData': actionData
        }))
    
    def schedule_report(self, reportData: bytes):
        msg_id = self._get_next_message_id()
        self.outbound_queue.put(('report', {
            'msgId': msg_id,
            'reportData': reportData
        }))

    def add_callback(self, dapp_id: int, callback):
        if dapp_id not in self.callbacks:
            e3_logger.debug(f"Add first callback for dApp {dapp_id}")
            self.callbacks[dapp_id] = [callback]
        else:
            if callback not in self.callbacks[dapp_id]:
                e3_logger.debug(f"Add additional callback for dApp {dapp_id}")
                self.callbacks[dapp_id].append(callback)
            else:
                e3_logger.warning(f"Callback already registered for dApp {dapp_id}, skipping")

    def remove_callback(self, dapp_id: int, callback=None):
        if dapp_id in self.callbacks:
            if callback is None:
                # Remove all callbacks for this dApp
                e3_logger.debug(f"Remove all callbacks for dApp {dapp_id}")
                del self.callbacks[dapp_id]
            elif callback in self.callbacks[dapp_id]:
                # Remove specific callback
                e3_logger.debug(f"Remove specific callback for dApp {dapp_id}")
                self.callbacks[dapp_id].remove(callback)
                if not self.callbacks[dapp_id]:
                    # Remove the dApp entry if no callbacks remain
                    del self.callbacks[dapp_id]
            else:
                e3_logger.warning(f"Specific callback not found for dApp {dapp_id}")
        else:
            e3_logger.warning(f"No callbacks found for dApp {dapp_id}")

    def _get_next_message_id(self):
        """Generate next message ID in thread-safe manner"""
        with self._message_id_lock:
            msg_id = self._message_id_counter
            self._message_id_counter = (self._message_id_counter % 100) + 1  # Wrap at 100 as per ASN.1 spec
            return msg_id

    def terminate_connections(self):
        e3_logger.info("Stop event")
        self.stop_event.set()
        
        if hasattr(self, "inbound_process"):
            self.inbound_process.join()
        if hasattr(self, "outbound_process"):
            self.outbound_process.join()

        self.e3_connector.dispose()

    def __del__(self):
        if not self.stop_event.is_set():
           self.terminate_connections()