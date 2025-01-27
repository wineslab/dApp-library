#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "E3-PDU.h"
#include "E3-SetupRequest.h"
#include <E3-SetupResponse.h>
#include <E3-IndicationMessage.h>
#include <E3-ControlAction.h>

// Function to encode an E3 PDU
int encode_E3_PDU(E3_PDU_t *pdu, uint8_t **buffer, size_t *buffer_size) {
    if (pdu->present == E3_PDU_PR_setupRequest) {
        printf("Encoding setupRequest: ranIdentifier = %ld\n", pdu->choice.setupRequest->ranIdentifier);
        printf("ranFunctionsList count = %d\n", pdu->choice.setupRequest->ranFunctionsList.list.count);
        for (size_t i = 0; i < pdu->choice.setupRequest->ranFunctionsList.list.count; i++) {
            printf("ranFunction[%zu] = %ld\n", i, pdu->choice.setupRequest->ranFunctionsList.list.array[i][0]);
        }
    }
    else if (pdu->present == E3_PDU_PR_indicationMessage) {
        printf("Encoding indicationMessage:\n");
        printf("Message len = %ld\n", pdu->choice.indicationMessage->protocolData.size);
        for (size_t i = 0; i < pdu->choice.indicationMessage->protocolData.size; i++) {
            printf("protocolData[%zu] = %d\n", i, pdu->choice.indicationMessage->protocolData.buf[i]);
        }
    } else {
        printf("Unexpected PDU choice\n");
        return -1;
    }

    asn_enc_rval_t enc_rval = aper_encode_to_buffer(&asn_DEF_E3_PDU, NULL, pdu, *buffer, *buffer_size);
    if (enc_rval.encoded == -1) {
        fprintf(stderr, "APER encoding failed for type: %s\n", enc_rval.failed_type ? enc_rval.failed_type->name : "Unknown");
        return -1;
    }

    *buffer_size = enc_rval.encoded;
    return 0;
}

// Function to decode an E3 PDU
E3_PDU_t *decode_E3_PDU(uint8_t *buffer, size_t buffer_size) {
    asn_dec_rval_t dec_rval;

    // Ensure buffer size is reasonable
    if (buffer_size == 0) {
        fprintf(stderr, "Buffer size is 0, nothing to decode.\n");
        return NULL;
    }

    // Initialize PDU structure pointer to NULL
    E3_PDU_t *pdu = NULL;

    // Decode the buffer into the PDU structure
    dec_rval = aper_decode(0, &asn_DEF_E3_PDU, (void **)&pdu, buffer, buffer_size, 0, 0);
    if (dec_rval.code != RC_OK) {
        fprintf(stderr, "APER decoding failed with code %d\n", dec_rval.code);
        ASN_STRUCT_FREE(asn_DEF_E3_PDU, pdu); // Free if partially allocated
        return NULL;
    }

    return pdu;
}

long parse_setup_response(E3_SetupResponse_t *response){
    printf("Parsing setupResponse: responseCode = %ld\n", response->responseCode);
    if (response->responseCode == 0)
    {
        printf("Response is positive.\n");
    }
    else if (response->responseCode == 1)
    {
        printf("Response is negative.\n");
    }
    else
    {
        printf("Unknown response code.\n");
    }
    return response->responseCode;
}


uint8_t* parse_control_action(E3_ControlAction_t *controlAction){
    printf("Parsing Control Action\n");
    size_t actionDataSize = controlAction->actionData.size;
    uint8_t *actionData = (uint8_t *) calloc(actionDataSize, sizeof(uint8_t));
    for (int i = 0; i < actionDataSize;i++)
        actionData[i] = controlAction->actionData.buf[i];

    return actionData;
}

// Function to create an E3 Setup Request PDU
E3_PDU_t* create_setup_request(long ranIdentifier, long *ranFunctions, size_t ranFunctionsCount) {
    if (ranIdentifier < 0 || ranIdentifier > 15) {
        fprintf(stderr, "Invalid ranIdentifier: must be in range 0 to 15\n");
        return NULL;
    }

    E3_PDU_t *pdu = malloc(sizeof(E3_PDU_t));
    if (!pdu) {
        fprintf(stderr, "Failed to allocate memory for E3_PDU_t\n");
        return NULL;
    }

    memset(pdu, 0, sizeof(E3_PDU_t));
    pdu->present = E3_PDU_PR_setupRequest;

    pdu->choice.setupRequest = calloc(1, sizeof(E3_SetupRequest_t));
    if (!pdu->choice.setupRequest) {
        fprintf(stderr, "Failed to allocate memory for E3_SetupRequest_t\n");
        free(pdu);
        return NULL;
    }

    pdu->choice.setupRequest->ranIdentifier = ranIdentifier;

    pdu->choice.setupRequest->ranFunctionsList.list.count = ranFunctionsCount;
    pdu->choice.setupRequest->ranFunctionsList.list.size = ranFunctionsCount * sizeof(long);
    pdu->choice.setupRequest->ranFunctionsList.list.array = malloc(pdu->choice.setupRequest->ranFunctionsList.list.size);

    if (!pdu->choice.setupRequest->ranFunctionsList.list.array) {
        fprintf(stderr, "Failed to allocate memory for ranFunctionsList array\n");
        free(pdu->choice.setupRequest);
        free(pdu);
        return NULL;
    }

    for (size_t i = 0; i < ranFunctionsCount; i++) {
        pdu->choice.setupRequest->ranFunctionsList.list.array[i] = malloc(sizeof(long *));
        pdu->choice.setupRequest->ranFunctionsList.list.array[i][0] = ranFunctions[i];
    }

    return pdu;
}

// Function to create an E3 Indication Message
E3_PDU_t* create_indication_message(const int32_t *payload, size_t payload_length) {
    E3_PDU_t *pdu = malloc(sizeof(E3_PDU_t));
    if (!pdu) {
        printf("Failed to allocate memory for E3_PDU\n");
        return NULL;
    }

    memset(pdu, 0, sizeof(E3_PDU_t));
    pdu->present = E3_PDU_PR_indicationMessage;
    pdu->choice.indicationMessage = calloc(1, sizeof(E3_IndicationMessage_t));
    if (!pdu->choice.indicationMessage) {
        printf("Failed to allocate memory for E3_IndicationMessage_t\n");
        free(pdu);
        return NULL;
    }

    pdu->choice.indicationMessage->protocolData.buf = malloc(payload_length);
    if (!pdu->choice.indicationMessage->protocolData.buf) {
        printf("Failed to allocate memory for protocolData\n");
        ASN_STRUCT_FREE(asn_DEF_E3_PDU, pdu);
        return NULL;
    }
    memcpy(pdu->choice.indicationMessage->protocolData.buf, payload, payload_length);
    pdu->choice.indicationMessage->protocolData.size = payload_length;

    return pdu;
}

// Function to free an E3 PDU
void free_E3_PDU(E3_PDU_t *pdu) {
    ASN_STRUCT_FREE(asn_DEF_E3_PDU, pdu);
}
