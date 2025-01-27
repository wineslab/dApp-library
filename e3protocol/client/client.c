#include <stdio.h>
#include <string.h>
#include <arpa/inet.h>
#include <netinet/sctp.h>
#include <unistd.h>
#include "e3.h"

#define SERVER_PORT 5000
#define BUFFER_SIZE 2048

void print_buffer(uint8_t *buffer, size_t size_to_read)
{
    // check dimension outside
    for (int i = 0; i < size_to_read; i++)
        printf("%02x", buffer[i]);

    printf("\n");
}

int main()
{
    int sockfd;
    struct sockaddr_in servaddr;
    uint8_t *buffer = malloc(BUFFER_SIZE);
    size_t buffer_size = BUFFER_SIZE;
    size_t payload = 0;

    if (!buffer)
    {
        fprintf(stderr, "Failed to allocate buffer\n");
        return -1;
    }

    sockfd = socket(AF_INET, SOCK_STREAM, IPPROTO_SCTP);
    if (sockfd < 0)
    {
        perror("Socket creation failed");
        free(buffer);
        return -1;
    }

    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = inet_addr("127.0.0.1");
    servaddr.sin_port = htons(SERVER_PORT);

    if (connect(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0)
    {
        perror("Connection failed");
        free(buffer);
        close(sockfd);
        return -1;
    }

    size_t ranFunctionsCount = 3;
    long *ranFunctions = calloc(ranFunctionsCount, sizeof(long));
    ranFunctions[0] = 101;
    ranFunctions[1] = 102;
    ranFunctions[2] = 103;
    E3_PDU_t *setupRequest = create_setup_request(1, ranFunctions, ranFunctionsCount);
    if (!setupRequest)
    {
        fprintf(stderr, "Failed to create setup request\n");
        free(buffer);
        close(sockfd);
        return -1;
    }

    if (encode_E3_PDU(setupRequest, &buffer, &buffer_size) == 0)
    {
        payload += buffer_size;
        send(sockfd, buffer, buffer_size, 0);
    }
    else
    {
        fprintf(stderr, "Failed to encode PDU\n");
    }

    free_E3_PDU(setupRequest);

    // Wait for the response
    size_t ret = recv(sockfd, buffer, buffer_size, 0);
    payload += ret;
    print_buffer(buffer, ret);

    E3_PDU_t *setupResponse = decode_E3_PDU(buffer, ret);

    // Check which CHOICE is present and process accordingly
    long res = 0;
    if (setupResponse->present == E3_PDU_PR_setupResponse)
    {
        res = parse_setup_response(setupResponse->choice.setupResponse);
        printf("Response Code: %ld\n", res);
    }
    else
    {
        printf("Unexpected PDU choice\n");
        res = 1;
    }

    free_E3_PDU(setupResponse);

    if (res == 0)
    {
        for (int i = 0; i < 200; i++)
        {
            // Create indication message
            size_t message_len = 10;
            int32_t *message = (int32_t *)malloc(sizeof(int32_t) * message_len);
            for (int i = 0; i < message_len; i++)
            {
                message[i] = i;
            }
            E3_PDU_t *indicationMessage = create_indication_message(message, message_len);

            if (encode_E3_PDU(indicationMessage, &buffer, &buffer_size) == 0)
            {
                payload += buffer_size;
                send(sockfd, buffer, buffer_size, 0);
            }
            else
            {
                fprintf(stderr, "Failed to encode PDU\n");
            }

            free_E3_PDU(indicationMessage);

            // Only for test purpose we receive the control Action
            ret = recv(sockfd, buffer, buffer_size, 0);
            payload += ret;

            print_buffer(buffer, ret);

            E3_PDU_t *controlAction = decode_E3_PDU(buffer, ret);

            // Check which CHOICE is present and process accordingly

            if (controlAction->present == E3_PDU_PR_controlAction)
            {
                // This code can be improved, especially the internal function
                u_int8_t *action_list = parse_control_action(controlAction->choice.controlAction);
                size_t action_list_size = controlAction->choice.controlAction->actionData.size;
                for (size_t i = 0; i < action_list_size; i++)
                    printf("action_list[%zu] = %u\n", i, action_list[i]);
            }
            else
            {
                printf("Unexpected PDU choice instead of control\n");
            }

            free_E3_PDU(controlAction);
        }
    }
    else
    {
        printf("Negative setup response \n");
    }

    printf("Payload: %ld\n", payload);

    free(buffer);
    close(sockfd);
    return 0;
}
