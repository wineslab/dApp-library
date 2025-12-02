import os
from abc import ABC, abstractmethod
import asn1tools


class E3Encoder(ABC):
    """Abstract base class for E3 message encoding/decoding"""
    
    def __init__(self):
        self.encoding_type = "unknown"
    
    @abstractmethod
    def encode_pdu(self, pdu_type: str, pdu_data: dict) -> bytes:
        """Encode a PDU message to bytes"""
        pass
    
    @abstractmethod
    def decode_pdu(self, data: bytes) -> tuple:
        """Decode bytes to PDU message, returns (pdu_type, pdu_data)"""
        pass
    
    @abstractmethod
    def create_setup_request(self, dappId: int = 1, ranFunctions: list = [], msgId: int = 1, actionType: str = "insert") -> bytes:
        """Create a setup request message"""
        pass
    
    @abstractmethod
    def create_subscription_request(self,  msgId: int = 1, dappId: int = 1, actionType: str = "insert", ranFunctionId: int = 1) -> bytes:
        """Create a subscription request message"""
        pass
    
    @abstractmethod
    def create_message_ack(self, msgId: int, requestId: int, responseCode: str = "positive") -> bytes:
        """Create a message acknowledgment"""
        pass
    
    @abstractmethod
    def create_control_action(self, msgId: int, dappId: int = 1, ranFunctionId: int = 1, actionData: bytes = b"") -> bytes:
        """Create a control action message"""
        pass
    
    @abstractmethod
    def create_dapp_report(self, msgId: int, reportData: bytes) -> bytes:
        """Create a dapp report message"""
        pass
    
    @abstractmethod
    def create_indication_message(self, msgId: int, protocolData: bytes) -> bytes:
        """Create an indication message"""
        pass


class AsnE3Encoder(E3Encoder):
    """ASN.1 implementation of E3 message encoding/decoding"""
    
    def __init__(self, asn_file_path: str = None):
        super().__init__()
        self.encoding_type = "asn1_per"
        
        # Load ASN.1 definitions
        if asn_file_path is None:
            asn_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./defs/e3.asn")
        
        self.defs = asn1tools.compile_files(asn_file_path, codec="per")
    
    def encode_pdu(self, pdu_type: str, pdu_data: dict) -> bytes:
        """Encode a PDU message to bytes using ASN.1 PER"""
        pdu_message = (pdu_type, pdu_data)
        return self.defs.encode("E3-PDU", pdu_message)
    
    def decode_pdu(self, data: bytes) -> tuple:
        """Decode bytes to PDU message using ASN.1 PER, returns (pdu_type, pdu_data)"""
        return self.defs.decode('E3-PDU', data)
    
    def create_setup_request(self, dappId: int = 1, ranFunctions: list = [], msgId: int = 1, actionType: str = "insert") -> bytes:
        """Create a setup request message"""
        setup_request_data = {
            "id": msgId,
            "dAppIdentifier": dappId, 
            "ranFunctionList": ranFunctions,
            "type": actionType
        }
        return self.encode_pdu("setupRequest", setup_request_data)
    
    def create_subscription_request(self,  msgId: int = 1, dappId: int = 1, actionType: str = "insert", ranFunctionId: int = 1) -> bytes:
        """Create a subscription request message"""
        subscription_request_data = {
            "id": msgId,
            "dAppIdentifier": dappId,
            "type": actionType,
            "ranFunctionIdentifier": ranFunctionId
        }
        return self.encode_pdu("subscriptionRequest", subscription_request_data)
    
    def create_message_ack(self, msgId: int, requestId: int, responseCode: str = "positive") -> bytes:
        """Create a message acknowledgment"""
        message_ack_data = {
            "id": msgId,
            "requestId": requestId,
            "responseCode": responseCode
        }
        return self.encode_pdu("messageAck", message_ack_data)
    
    def create_control_action(self, msgId: int, dappId: int = 1, ranFunctionId: int = 1, actionData: bytes = b"") -> bytes:
        """Create a control action message"""
        control_action_data = {
            "id": msgId,
            "dAppIdentifier": dappId,
            "ranFunctionIdentifier": ranFunctionId,
            "actionData": actionData
        }
        return self.encode_pdu("controlAction", control_action_data)
    
    def create_indication_message(self, msgId: int, protocolData: bytes) -> bytes:
        """Create an indication message"""
        indication_message_data = {
            "id": msgId,
            "protocolData": protocolData
        }
        return self.encode_pdu("indicationMessage", indication_message_data)
    
    def create_dapp_report(self, msgId: int, dappId: int, ranFunctionId: int, reportData: bytes) -> bytes:
        """Create a dApp report message"""
        dapp_report_data = {
            "id": msgId,
            "dAppIdentifier": dappId,
            "ranFunctionIdentifier": ranFunctionId,
            "reportData": reportData
        }
        return self.encode_pdu("dAppReport", dapp_report_data)