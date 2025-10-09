#include "E3-PDU.h"
#include "E3-SetupResponse.h"
#include "E3-SubscriptionResponse.h"
#include "E3-MessageAck.h"
#include <E3-ControlAction.h>

// Setup and teardown functions
E3_PDU_t *create_setup_request(long msgId, int dappIdentifier, long *ranFunctions, size_t ranFunctionsCount, long actionType);
E3_PDU_t *create_indication_message(long msgId, int32_t *payload, size_t payloadCount);

// Subscription functions
E3_PDU_t *create_subscription_request(long msgId, long actionType, long ranFunctionId);
E3_PDU_t *create_message_ack(long msgId, long requestId, long responseCode);

// Encoding/decoding functions
int encode_E3_PDU(E3_PDU_t *pdu, uint8_t **buffer, size_t *buffer_size);
E3_PDU_t *decode_E3_PDU(uint8_t *buffer, size_t buffer_size);
void free_E3_PDU(E3_PDU_t *pdu);

// Parsing functions
long parse_setup_response(E3_SetupResponse_t *response);
long parse_subscription_response(E3_SubscriptionResponse_t *response);
long parse_message_ack(E3_MessageAck_t *ack);
uint8_t *parse_control_action(E3_ControlAction_t *controlAction);