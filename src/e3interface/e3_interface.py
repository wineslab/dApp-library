import os
import multiprocessing
import queue
import threading
from .e3_connector import E3Connector
from .e3_logging import e3_logger
import asn1tools

class E3Interface:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(E3Interface, cls).__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "initialized"):
            self.callbacks = []
            self.stop_event = multiprocessing.Event()
            self.initialized = True

            # Create an E3Connector instance based on the configuration
            self.e3_connector = E3Connector.setup_connector(kwargs.get('link', ''), kwargs.get('transport', '')) 
            
            e3_logger.info(f"Endpoint setup {self.e3_connector.setup_endpoint}")
            e3_logger.info(f"Endpoint inbound {self.e3_connector.inbound_endpoint}")
            e3_logger.info(f"Endpoint outbound {self.e3_connector.outbound_endpoint}")

            self.defs = asn1tools.compile_files(os.path.join(os.path.dirname(os.path.realpath(__file__)), "./defs/e3.asn"), codec="per") 
            self.outbound_queue = multiprocessing.Queue()
    
    def send_setup_request(self, dappId: int = 1) -> bool:
        e3_logger.info("Start setup request")
        payload = self.create_setup_request(dappId)
        try:
            response = self.e3_connector.send_setup_request(payload)
        except ConnectionRefusedError as e:
            e3_logger.error(f"Unable to connect to E3 setup endpoint, connection refused: {e}")
            return False

        e3_logger.info("Setup response received")
        pdu = self.defs.decode('E3-PDU', response)
        if pdu[0] == "setupResponse":
            e3_setup_response = pdu[1]
            e3_logger.info(e3_setup_response)
            # Either here or in setup connections we need E3 sub request and response
            self.setup_connections()
            return e3_setup_response['responseCode'] == 'positive'
        else:
            return False

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
                e3_logger.debug(data.hex())
                pdu = self.defs.decode("E3-PDU", data)
                e3_logger.debug(f"Data decoded")
                match pdu[0]:
                    case "indicationMessage":
                        e3_indication_message = pdu[1]
                        protocolData = e3_indication_message['protocolData']
                        e3_logger.debug(protocolData)
                        e3_logger.debug(f"Indication message protocolData {len(protocolData)}")
                        self._handle_incoming_data(protocolData)

                    case "xAppControlAction":
                        e3_xapp_control_action = pdu[1]
                        raise NotImplementedError()

                    case _:
                        raise ValueError("Unrecognized value ", pdu)
                    
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
                        payload = self.create_control_action(data)
                    case "report":
                        payload = self.create_dapp_report(data)

                    case _:
                        raise ValueError("Unrecognized value ", msg)

                e3_logger.debug(f"Send the pdu encoded {payload}")
                self.e3_connector.send(payload)

        except Exception as e:
            e3_logger.error(f"Error outbound thread: {e}")
            self.stop_event.set()
        finally:
            e3_logger.info("Close outbound connection")

    def _handle_incoming_data(self, data):
        for callback in self.callbacks:
            e3_logger.debug("Launch callback")
            callback(data)
                
    def schedule_control(self, payload: bytes):
        self.outbound_queue.put(('control', payload))
    
    def schedule_report(self, payload: bytes):
        self.outbound_queue.put(('report', payload))

    def add_callback(self, callback):
        if callback not in self.callbacks:
            e3_logger.debug("Add callback")
            self.callbacks.append(callback)

    def remove_callback(self, callback):
        if callback in self.callbacks:
            e3_logger.debug("Remove callback")
            self.callbacks.remove(callback)

    def terminate_connections(self):
        e3_logger.info("Stop event")
        self.stop_event.set()
        
        if hasattr(self, "inbound_process"):
            self.inbound_process.join()
        if hasattr(self, "outbound_process"):
            self.outbound_process.join()

        self.e3_connector.dispose()
  
    def create_control_action(self, actionData: bytes):
        control_message = ("controlAction", {"actionData": actionData})
        payload = self.defs.encode("E3-PDU", control_message)
        return payload

    def create_dapp_report(self, reportData: bytes):
        dapp_report_message = ("dAppReport", {"reportData": reportData})
        payload = self.defs.encode("E3-PDU", dapp_report_message)
        return payload
        
    def create_setup_request(self, ranId: int = 1, ranFunctions: list = []):
        setup_request_message = ("setupRequest", {"ranIdentifier": ranId, "ranFunctionsList": ranFunctions})
        payload = self.defs.encode("E3-PDU", setup_request_message)
        return payload

    def __del__(self):
        if not self.stop_event.is_set():
           self.terminate_connections()