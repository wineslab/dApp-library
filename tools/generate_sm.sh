#!/bin/bash

################################################################################
# E3 Service Model Generator Script
################################################################################
#
# Copyright (c) 2024-2025 Northeastern University
# Licensed under the Apache License, Version 2.0
#
# Author: Andrea Lacava <a.lacava@northeastern.edu>
#
# Description:
#   Generates skeleton code for new E3 Service Models on both agent and dApp
#   sides. This script automates the creation of boilerplate code including:
#   - Agent-side C implementation with ASN.1 encoding/decoding
#   - dApp-side Python implementation with E3 interface
#   - CMake build configuration
#   - ASN.1 schema integration
#
# References:
#   - E3AP Protocol: Extension of O-RAN E2AP for experimentation
#   - FlexRIC gen_sm.py: Original inspiration for SM generation script
#   - Spectrum SM: Reference implementation in this codebase
#
# Acknowledgments:
#   This work was partially supported by OUSD(R&E) through Army Research
#   Laboratory Cooperative Agreement Number W911NF-24-2-0065. The views and
#   conclusions contained in this document are those of the authors and should
#   not be interpreted as representing the official policies, either expressed
#   or implied, of the Army Research Laboratory or the U.S. Government. The
#   U.S. Government is authorized to reproduce and distribute reprints for
#   Government purposes notwithstanding any copyright notation herein. The work
#   was also partially supported by SERICS (PE00000014) 5GSec project, CUP
#   B53C22003990006, under the MUR National Recovery and Resilience Plan funded
#   by the European Union - NextGenerationEU, and by the U.S. National Science
#   Foundation under grants CNS-1925601, CNS-2117814, and CNS-2112471.
#.
#
# Usage:
#   ./generate_sm.sh --interactive
#   ./generate_sm.sh -n <name> -a <asn_file> [-r <ran_function_ids>] [-t <target>]
#
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAPP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# AGENT_ROOT can be overridden via environment variable or command line
AGENT_ROOT="${AGENT_ROOT:-$(cd "$DAPP_ROOT/../spear-openairinterface5g" && pwd 2>/dev/null || echo "")}"

# Configuration variables
SM_NAME=""
ASN_FILE=""
TARGET="both"
RAN_FUNCTION_IDS=""
FORMAT="asn1"
INTERACTIVE=false
AGENT_ROOT_OVERRIDE=""

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Generate E3 Service Model skeleton code"
    echo ""
    echo "Options:"
    echo "  -n, --name NAME          Service Model name (required)"
    echo "  -a, --asn-file FILE      ASN.1 definition file (required)"
    echo "  -t, --target TARGET      Target: agent, dapp, or both (default: both)"
    echo "  -r, --ran-functions IDS  Comma-separated RAN function IDs (e.g., 2,3,4)"
    echo "  -f, --format FORMAT      Encoding format: asn1 or json (default: asn1)"
    echo "  --agent-root PATH        Path to OAI agent repository root"
    echo "  -i, --interactive        Interactive mode with prompts"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  AGENT_ROOT              Path to OAI agent repository (can also use --agent-root)"
    echo ""
    echo "Examples:"
    echo "  $0 --interactive"
    echo "  $0 -n positioning -a positioning.asn -r 2 -t both"
    echo "  $0 -n kpm -a kpm.asn -r 3,4 -f json --agent-root /path/to/oai"
    exit 1
}

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Interactive prompts
prompt_interactive() {
    echo -e "${BLUE}=== E3 Service Model Generator ===${NC}"
    echo ""
    
    # SM name
    while [[ -z "$SM_NAME" ]]; do
        read -p "Service Model name (e.g., positioning, kpm): " SM_NAME
        if [[ ! "$SM_NAME" =~ ^[a-zA-Z][a-zA-Z0-9_]*$ ]]; then
            log_error "Invalid SM name. Use alphanumeric characters and underscores only."
            SM_NAME=""
        fi
    done
    
    # ASN.1 file
    while [[ -z "$ASN_FILE" || ! -f "$ASN_FILE" ]]; do
        read -p "ASN.1 definition file path: " ASN_FILE
        if [[ ! -f "$ASN_FILE" ]]; then
            log_error "File not found: $ASN_FILE"
            ASN_FILE=""
        fi
    done
    
    # Target
    echo ""
    echo "Target options:"
    echo "  1) agent  - Only generate agent-side code"
    echo "  2) dapp   - Only generate dApp-side code"  
    echo "  3) both   - Generate both sides (recommended)"
    read -p "Select target [1-3] (default: 3): " target_choice
    case $target_choice in
        1) TARGET="agent" ;;
        2) TARGET="dapp" ;;
        3|"") TARGET="both" ;;
        *) log_warning "Invalid choice, using 'both'"; TARGET="both" ;;
    esac
    
    # RAN function IDs
    read -p "RAN function IDs (comma-separated, e.g., 2,3,4): " RAN_FUNCTION_IDS
    
    # Format
    echo ""
    echo "Encoding format:"
    echo "  1) asn1 - ASN.1 encoding (recommended)"
    echo "  2) json - JSON encoding"
    read -p "Select format [1-2] (default: 1): " format_choice
    case $format_choice in
        1|"") FORMAT="asn1" ;;
        2) FORMAT="json" ;;
        *) log_warning "Invalid choice, using 'asn1'"; FORMAT="asn1" ;;
    esac
    
    # Agent root path (if targeting agent)
    if [[ "$TARGET" == "agent" || "$TARGET" == "both" ]]; then
        echo ""
        if [[ -n "$AGENT_ROOT" && -d "$AGENT_ROOT" ]]; then
            log_info "Detected AGENT_ROOT: $AGENT_ROOT"
            read -p "Use this path? [Y/n]: " use_detected
            if [[ "$use_detected" =~ ^[Nn]$ ]]; then
                read -p "Enter OAI repository path: " AGENT_ROOT_OVERRIDE
            fi
        else
            read -p "Enter OAI repository path: " AGENT_ROOT_OVERRIDE
        fi
    fi
    
    echo ""
    log_info "Configuration:"
    log_info "  SM Name: $SM_NAME"
    log_info "  ASN.1 File: $ASN_FILE"
    log_info "  Target: $TARGET"
    log_info "  RAN Functions: $RAN_FUNCTION_IDS"
    log_info "  Format: $FORMAT"
    if [[ "$TARGET" == "agent" || "$TARGET" == "both" ]]; then
        log_info "  Agent Root: ${AGENT_ROOT_OVERRIDE:-$AGENT_ROOT}"
    fi
    log_info "  dApp Root: $DAPP_ROOT"
    echo ""
    
    read -p "Continue with generation? [Y/n]: " confirm
    if [[ "$confirm" =~ ^[Nn]$ ]]; then
        log_info "Generation cancelled"
        exit 0
    fi
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--name)
                SM_NAME="$2"
                shift 2
                ;;
            -a|--asn-file)
                ASN_FILE="$2"
                shift 2
                ;;
            -t|--target)
                TARGET="$2"
                shift 2
                ;;
            -r|--ran-functions)
                RAN_FUNCTION_IDS="$2"
                shift 2
                ;;
            -f|--format)
                FORMAT="$2"
                shift 2
                ;;
            --agent-root)
                AGENT_ROOT_OVERRIDE="$2"
                shift 2
                ;;
            -i|--interactive)
                INTERACTIVE=true
                shift
                ;;
            -h|--help)
                usage
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done
}

# Validate inputs
validate_inputs() {
    # Override AGENT_ROOT if specified via command line
    if [[ -n "$AGENT_ROOT_OVERRIDE" ]]; then
        AGENT_ROOT="$AGENT_ROOT_OVERRIDE"
    fi
    
    if [[ -z "$SM_NAME" ]]; then
        log_error "Service Model name is required"
        exit 1
    fi
    
    if [[ ! "$SM_NAME" =~ ^[a-zA-Z][a-zA-Z0-9_]*$ ]]; then
        log_error "Invalid SM name. Use alphanumeric characters and underscores only."
        exit 1
    fi
    
    if [[ -z "$ASN_FILE" ]]; then
        log_error "ASN.1 file is required"
        exit 1
    fi
    
    if [[ ! -f "$ASN_FILE" ]]; then
        log_error "ASN.1 file not found: $ASN_FILE"
        exit 1
    fi
    
    if [[ "$TARGET" != "agent" && "$TARGET" != "dapp" && "$TARGET" != "both" ]]; then
        log_error "Invalid target: $TARGET. Must be 'agent', 'dapp', or 'both'"
        exit 1
    fi
    
    if [[ "$FORMAT" != "asn1" && "$FORMAT" != "json" ]]; then
        log_error "Invalid format: $FORMAT. Must be 'asn1' or 'json'"
        exit 1
    fi
    
    # Validate AGENT_ROOT if agent target is selected
    if [[ "$TARGET" == "agent" || "$TARGET" == "both" ]]; then
        if [[ -z "$AGENT_ROOT" || ! -d "$AGENT_ROOT" ]]; then
            log_error "AGENT_ROOT not found or invalid: $AGENT_ROOT"
            log_error "Please specify the OAI repository path via:"
            log_error "  - Command line: --agent-root /path/to/oai"
            log_error "  - Environment: export AGENT_ROOT=/path/to/oai"
            exit 1
        fi
        
        if [[ ! -d "$AGENT_ROOT/openair2/E3AP" ]]; then
            log_error "Invalid AGENT_ROOT: E3AP directory not found"
            log_error "Expected: $AGENT_ROOT/openair2/E3AP"
            exit 1
        fi
    fi
}

# Convert RAN function IDs to C array format
format_ran_function_ids() {
    if [[ -z "$RAN_FUNCTION_IDS" ]]; then
        echo "2" # Default RAN function ID
        return
    fi
    
    echo "$RAN_FUNCTION_IDS" | tr ',' ' '
}

# Generate agent-side code
generate_agent_code() {
    local sm_name="$1"
    local sm_upper="$(echo "$sm_name" | tr '[:lower:]' '[:upper:]')"
    local agent_sm_dir="$AGENT_ROOT/openair2/E3AP/service_models/${sm_name}_sm"
    
    log_info "Generating agent-side code for $sm_name SM..."
    
    # Create directory structure
    mkdir -p "$agent_sm_dir"
    mkdir -p "$agent_sm_dir/MESSAGES/ASN1/V1"
    
    # Copy ASN.1 file
    cp "$ASN_FILE" "$agent_sm_dir/MESSAGES/ASN1/V1/"
    local asn_filename="$(basename "$ASN_FILE")"
    
    # Generate e3sm_${sm_name}.cmake file
    cat > "$agent_sm_dir/MESSAGES/ASN1/V1/e3sm_${sm_name}.cmake" << 'CMAKEFILE'
set(${sm_upper}_SM_GRAMMAR ASN1/V1/${asn_filename})

set(${sm_name}_sm_source
    ANY_aper.c
    ANY.c
    ANY_uper.c
    ANY_xer.c
    aper_decoder.c
    aper_encoder.c
    aper_opentype.c
    aper_support.c
    asn_application.c
    asn_bit_data.c
    asn_codecs_prim.c
    asn_codecs_prim_xer.c
    asn_internal.c
    asn_random_fill.c
    ber_tlv_length.c
    ber_tlv_tag.c
    constr_CHOICE.c
    constr_SEQUENCE.c
    constr_TYPE.c
    constraints.c
    OCTET_STRING_aper.c
    OCTET_STRING.c
    OCTET_STRING_uper.c
    OCTET_STRING_xer.c
    per_decoder.c
    per_encoder.c
    per_opentype.c
    per_support.c
    BOOLEAN.c
    BOOLEAN_aper.c
    BOOLEAN_uper.c
    BOOLEAN_xer.c
    INTEGER.c
    INTEGER_aper.c
    INTEGER_uper.c
    INTEGER_xer.c
    uper_decoder.c
    uper_encoder.c
    uper_opentype.c
    uper_support.c
    xer_decoder.c
    xer_encoder.c
    xer_support.c
    # TODO: Add ${sm_name} SM specific generated files here
    # Example: ${sm_name^}-MessageType.c
)
CMAKEFILE
    
    # Generate CMakeLists for ASN.1
    cat > "$agent_sm_dir/MESSAGES/CMakeLists.txt" << EOF
# Generated CMakeLists.txt for ${sm_name} SM ASN.1

set(${sm_upper}_SM_VERSION 1 0 0)
make_version(${sm_upper}_SM_cc \${${sm_upper}_SM_VERSION})
string(REPLACE ";" "." ${sm_upper}_SM_RELEASE "\${${sm_upper}_SM_VERSION}")

if(${sm_upper}_SM_RELEASE VERSION_EQUAL "1.0.0")
  include(ASN1/V1/e3sm_${sm_name}.cmake)
else()
  message(FATAL_ERROR "unknown ${sm_upper}_SM_RELEASE \${${sm_upper}_SM_RELEASE}")
endif()

message(STATUS "Selected ${sm_upper}_SM_VERSION: \${${sm_upper}_SM_RELEASE}")

add_custom_command(OUTPUT \${${sm_name}_sm_source} \${${sm_name}_sm_headers}
  COMMAND \${ASN1C_EXEC} -pdu=all -gen-APER -gen-UPER -no-gen-JER -no-gen-BER -no-gen-OER -fno-include-deps -fcompound-names -findirect-choice -no-gen-example -D \${CMAKE_CURRENT_BINARY_DIR} \${CMAKE_CURRENT_SOURCE_DIR}/\${${sm_upper}_SM_GRAMMAR}
  DEPENDS \${CMAKE_CURRENT_SOURCE_DIR}/\${${sm_upper}_SM_GRAMMAR}
  COMMENT "Generating ${sm_name} SM source files from \${CMAKE_CURRENT_SOURCE_DIR}/\${${sm_upper}_SM_GRAMMAR}"
)

add_library(${sm_name}_sm_asn1 \${${sm_name}_sm_source})
target_include_directories(${sm_name}_sm_asn1 PUBLIC "\${CMAKE_CURRENT_BINARY_DIR}")
target_compile_options(${sm_name}_sm_asn1 PRIVATE -DASN_DISABLE_OER_SUPPORT -w)

# Export variables to parent scope
set(${sm_upper}_SM_ASN1_LIB ${sm_name}_sm_asn1 PARENT_SCOPE)
set(${sm_upper}_SM_ASN1_INCLUDE_DIR "\${CMAKE_CURRENT_BINARY_DIR}" PARENT_SCOPE)
EOF

    # Generate main SM header
    local ran_func_array="$(format_ran_function_ids)"
    cat > "$agent_sm_dir/${sm_name}_sm.h" << EOF
#ifndef ${sm_upper}_SM_H
#define ${sm_upper}_SM_H

#include "../sm_interface.h"

// Forward declarations for ${sm_name} SM functions
int ${sm_name}_sm_init(e3_service_model_t *sm);
int ${sm_name}_sm_destroy(e3_service_model_t *sm);
void* ${sm_name}_sm_thread_main(void *context);
int ${sm_name}_sm_process_dapp_control_action(uint32_t ran_function_id, uint8_t *encoded_data, size_t size);

// ${sm_name} SM RAN function IDs
$(echo "$ran_func_array" | awk '{for(i=1;i<=NF;i++) printf "#define %s_SM_RAN_FUNCTION_ID_%d %s\n", toupper("'$sm_name'"), i, $i}')

// ${sm_name} SM specific structures
typedef struct {
    // TODO: Add SM-specific context fields
    bool initialized;
} ${sm_name}_sm_context_t;

// Export the ${sm_name} SM instance
extern e3_service_model_t ${sm_name}_sm;

#endif // ${sm_upper}_SM_H
EOF

    # Generate main SM implementation
    cat > "$agent_sm_dir/${sm_name}_sm.c" << EOF
#include "${sm_name}_sm.h"
#include "${sm_name}_enc.h"
#include "${sm_name}_dec.h"

#include "common/utils/LOG/log.h"
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <sys/time.h>

// ${sm_name} SM RAN function configuration
static uint32_t ${sm_name}_ran_functions[] = {$(echo "$ran_func_array" | tr ' ' ',')};

// ${sm_name} SM instance
e3_service_model_t ${sm_name}_sm = {
    .name = "${sm_name}_sm",
    .version = 1,
    .ran_function_ids = ${sm_name}_ran_functions,
    .ran_function_count = sizeof(${sm_name}_ran_functions) / sizeof(${sm_name}_ran_functions[0]),
    .init = ${sm_name}_sm_init,
    .destroy = ${sm_name}_sm_destroy,
    .thread_main = ${sm_name}_sm_thread_main,
    .process_dapp_control_action = ${sm_name}_sm_process_dapp_control_action,
    .thread_context = NULL,
    .is_running = false,
#ifdef ${sm_upper}_SM_ASN1_FORMAT
    .format = FORMAT_ASN1
#else
    .format = FORMAT_JSON
#endif
};

// Global SM context
static ${sm_name}_sm_context_t *${sm_name}_context = NULL;

/**
 * Initialize the ${sm_name} SM
 */
int ${sm_name}_sm_init(e3_service_model_t *sm) {
    if (${sm_name}_context) {
        LOG_W(E3AP, "${sm_name} SM already initialized\\n");
        return SM_SUCCESS;
    }
    
    ${sm_name}_context = malloc(sizeof(${sm_name}_sm_context_t));
    if (!${sm_name}_context) {
        LOG_E(E3AP, "Failed to allocate ${sm_name} SM context\\n");
        return SM_ERROR_MEMORY;
    }
    
    memset(${sm_name}_context, 0, sizeof(${sm_name}_sm_context_t));
    
    // TODO: Add SM-specific initialization
    ${sm_name}_context->initialized = true;
    
    LOG_I(E3AP, "${sm_name} SM initialized successfully\\n");
    return SM_SUCCESS;
}

/**
 * Destroy the ${sm_name} SM
 */
int ${sm_name}_sm_destroy(e3_service_model_t *sm) {
    if (!${sm_name}_context) {
        return SM_SUCCESS;
    }
    
    // TODO: Add SM-specific cleanup
    
    free(${sm_name}_context);
    ${sm_name}_context = NULL;
    
    LOG_I(E3AP, "${sm_name} SM destroyed\\n");
    return SM_SUCCESS;
}

/**
 * Get current timestamp in microseconds
 */
static uint64_t get_current_timestamp_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

/**
 * Main thread function for ${sm_name} SM
 */
void* ${sm_name}_sm_thread_main(void *context) {
    sm_thread_context_t *thread_ctx = (sm_thread_context_t *)context;
    ${sm_name}_sm_context_t *sm_ctx = ${sm_name}_context;
    
    if (!sm_ctx || !sm_ctx->initialized) {
        LOG_E(E3AP, "${sm_name} SM context not initialized\\n");
        return NULL;
    }
    
    LOG_I(E3AP, "${sm_name} SM thread started\\n");
    
    // TODO: Add SM-specific thread initialization
    
    // Main processing loop
    while (1) {
        // Check if we should stop
        pthread_mutex_lock(&thread_ctx->stop_mutex);
        bool should_stop = thread_ctx->should_stop;
        pthread_mutex_unlock(&thread_ctx->stop_mutex);
        
        if (should_stop) {
            LOG_I(E3AP, "${sm_name} SM thread stopping\\n");
            break;
        }
        
        // TODO: Add SM-specific data processing
        // Example:
        // 1. Collect data from RAN/T-tracer
        // 2. Process the data
        // 3. Encode indication message
        // 4. Update shared indication data
        
        // For now, just sleep to avoid busy waiting
        usleep(100000); // 100ms
    }
    
    LOG_I(E3AP, "${sm_name} SM thread finished\\n");
    return NULL;
}

/**
 * Process control messages for ${sm_name} SM
 */
int ${sm_name}_sm_process_dapp_control_action(uint32_t ran_function_id, uint8_t *encoded_data, size_t size) {
    // Validate RAN function ID
    bool valid_ran_function = false;
    for (uint32_t i = 0; i < ${sm_name}_sm.ran_function_count; i++) {
        if (${sm_name}_sm.ran_function_ids[i] == ran_function_id) {
            valid_ran_function = true;
            break;
        }
    }
    
    if (!valid_ran_function) {
        LOG_E(E3AP, "Invalid RAN function ID %u for ${sm_name} SM\\n", ran_function_id);
        return SM_ERROR_INVALID_PARAM;
    }
    
    if (!encoded_data || size == 0) {
        LOG_E(E3AP, "Invalid control data parameters\\n");
        return SM_ERROR_INVALID_PARAM;
    }
    
    LOG_I(E3AP, "${sm_name} SM: Processing control message (%zu bytes)\\n", size);
    
    // TODO: Add SM-specific control message processing
    // Example:
    // 1. Decode the control message using ${sm_name}_decode_control()
    // 2. Validate the control parameters
    // 3. Apply the control action
    // 4. Update SM state
    
    LOG_I(E3AP, "${sm_name} SM: Control action applied successfully\\n");
    
    return SM_SUCCESS;
}
EOF

    # Generate encoding header
    cat > "$agent_sm_dir/${sm_name}_enc.h" << EOF
#ifndef ${sm_upper}_ENC_H
#define ${sm_upper}_ENC_H

#include <stdint.h>
#include <stddef.h>

/**
 * ${sm_name} SM Encoding Functions
 * 
 * Provides encoding for ${sm_name} indication and control messages
 * with compile-time format selection (ASN.1 or JSON)
 */

// Compile-time format selection
#ifdef ${sm_upper}_SM_ASN1_FORMAT
    // TODO: Include generated ASN.1 headers
    // #include "${sm_name}-IndicationData.h"
    // #include "${sm_name}-ControlData.h"
#endif

#ifdef ${sm_upper}_SM_JSON_FORMAT
    #include <json-c/json.h>
#endif

/**
 * Encode ${sm_name} indication message
 * 
 * @param data SM-specific data structure
 * @param encoded_data Output buffer (allocated by function)
 * @param encoded_size Output size
 * @return 0 on success, negative on error
 */
int ${sm_name}_encode_indication(void *data, uint8_t **encoded_data, size_t *encoded_size);

/**
 * Encode ${sm_name} control message
 * 
 * @param control_data SM-specific control structure
 * @param encoded_data Output buffer (allocated by function)
 * @param encoded_size Output size
 * @return 0 on success, negative on error
 */
int ${sm_name}_encode_control(void *control_data, uint8_t **encoded_data, size_t *encoded_size);

#endif // ${sm_upper}_ENC_H
EOF

    # Generate encoding implementation
    cat > "$agent_sm_dir/${sm_name}_enc.c" << EOF
#include "${sm_name}_enc.h"
#include "../sm_interface.h"
#include "common/utils/LOG/log.h"
#include <stdlib.h>
#include <string.h>

#ifdef ${sm_upper}_SM_ASN1_FORMAT
/**
 * ASN.1 encoding implementation
 */

int ${sm_name}_encode_indication(void *data, uint8_t **encoded_data, size_t *encoded_size) {
    if (!data || !encoded_data || !encoded_size) {
        return SM_ERROR_INVALID_PARAM;
    }
    
    // TODO: Implement ASN.1 encoding for ${sm_name} indication
    LOG_E(E3AP, "${sm_name} ASN.1 indication encoding not implemented\\n");
    return SM_ERROR_INVALID_PARAM;
}

int ${sm_name}_encode_control(void *control_data, uint8_t **encoded_data, size_t *encoded_size) {
    if (!control_data || !encoded_data || !encoded_size) {
        return SM_ERROR_INVALID_PARAM;
    }
    
    // TODO: Implement ASN.1 encoding for ${sm_name} control
    LOG_E(E3AP, "${sm_name} ASN.1 control encoding not implemented\\n");
    return SM_ERROR_INVALID_PARAM;
}

#else
/**
 * JSON encoding implementation
 */

int ${sm_name}_encode_indication(void *data, uint8_t **encoded_data, size_t *encoded_size) {
    if (!data || !encoded_data || !encoded_size) {
        return SM_ERROR_INVALID_PARAM;
    }
    
    // TODO: Implement JSON encoding for ${sm_name} indication
    LOG_E(E3AP, "${sm_name} JSON indication encoding not implemented\\n");
    return SM_ERROR_INVALID_PARAM;
}

int ${sm_name}_encode_control(void *control_data, uint8_t **encoded_data, size_t *encoded_size) {
    if (!control_data || !encoded_data || !encoded_size) {
        return SM_ERROR_INVALID_PARAM;
    }
    
    // TODO: Implement JSON encoding for ${sm_name} control
    LOG_E(E3AP, "${sm_name} JSON control encoding not implemented\\n");
    return SM_ERROR_INVALID_PARAM;
}

#endif
EOF

    # Generate decoding header
    cat > "$agent_sm_dir/${sm_name}_dec.h" << EOF
#ifndef ${sm_upper}_DEC_H
#define ${sm_upper}_DEC_H

#include <stdint.h>
#include <stddef.h>

/**
 * ${sm_name} SM Decoding Functions
 * 
 * Provides decoding for ${sm_name} control messages
 * with compile-time format selection (ASN.1 or JSON)
 */

/**
 * Decode ${sm_name} control message
 * 
 * @param encoded_data Input encoded data
 * @param encoded_size Size of encoded data
 * @param decoded_control Output decoded control (allocated by function)
 * @return 0 on success, negative on error
 */
int ${sm_name}_decode_control(uint8_t *encoded_data, size_t encoded_size, 
                             void **decoded_control);

/**
 * Free decoded control structure
 * 
 * @param decoded_control Decoded control to free
 */
void ${sm_name}_free_decoded_control(void *decoded_control);

#endif // ${sm_upper}_DEC_H
EOF

    # Generate decoding implementation
    cat > "$agent_sm_dir/${sm_name}_dec.c" << EOF
#include "${sm_name}_dec.h"
#include "../sm_interface.h"  
#include "common/utils/LOG/log.h"
#include <stdlib.h>
#include <string.h>

#ifdef ${sm_upper}_SM_ASN1_FORMAT
/**
 * ASN.1 decoding implementation
 */

int ${sm_name}_decode_control(uint8_t *encoded_data, size_t encoded_size, 
                             void **decoded_control) {
    if (!encoded_data || encoded_size == 0 || !decoded_control) {
        return SM_ERROR_INVALID_PARAM;
    }
    
    // TODO: Implement ASN.1 decoding for ${sm_name} control
    LOG_E(E3AP, "${sm_name} ASN.1 control decoding not implemented\\n");
    return SM_ERROR_INVALID_PARAM;
}

#else
/**
 * JSON decoding implementation
 */

int ${sm_name}_decode_control(uint8_t *encoded_data, size_t encoded_size, 
                             void **decoded_control) {
    if (!encoded_data || encoded_size == 0 || !decoded_control) {
        return SM_ERROR_INVALID_PARAM;
    }
    
    // TODO: Implement JSON decoding for ${sm_name} control
    LOG_E(E3AP, "${sm_name} JSON control decoding not implemented\\n");
    return SM_ERROR_INVALID_PARAM;
}

#endif

/**
 * Common cleanup functions (format-independent)
 */

void ${sm_name}_free_decoded_control(void *decoded_control) {
    if (!decoded_control) return;
    
    // TODO: Add SM-specific cleanup
    free(decoded_control);
}
EOF

    # Generate SM CMakeLists.txt
    cat > "$agent_sm_dir/CMakeLists.txt" << EOF
# ${sm_name} Service Model CMakeLists.txt

cmake_minimum_required(VERSION 3.10)

# Add ASN.1 message generation
add_subdirectory(MESSAGES)

# ${sm_name} SM source files
set(${sm_upper}_SM_SOURCES
    ${sm_name}_sm.c
    ${sm_name}_enc.c
    ${sm_name}_dec.c
)

# Create ${sm_name} SM library
add_library(${sm_name}_sm STATIC \${${sm_upper}_SM_SOURCES})

# Include directories
target_include_directories(${sm_name}_sm PUBLIC
    \${CMAKE_CURRENT_SOURCE_DIR}
    \${CMAKE_CURRENT_SOURCE_DIR}/..
    \${CMAKE_CURRENT_BINARY_DIR}/MESSAGES  # ASN.1 generated headers
    \${CMAKE_SOURCE_DIR}/common/utils/LOG
    \${CMAKE_SOURCE_DIR}/openair2/E3AP
)

target_link_libraries(${sm_name}_sm ${sm_name}_sm_asn1)
add_dependencies(${sm_name}_sm ${sm_name}_sm_asn1)

# Compile-time format selection (inherit from E3AP)
if(E3_ENCODING_FORMAT STREQUAL "ASN1")
    target_compile_definitions(${sm_name}_sm PRIVATE ${sm_upper}_SM_ASN1_FORMAT)
    message(STATUS "${sm_name} SM: Using ASN.1 encoding")
else()
    target_compile_definitions(${sm_name}_sm PRIVATE ${sm_upper}_SM_JSON_FORMAT)
    target_link_libraries(${sm_name}_sm json-c)
    message(STATUS "${sm_name} SM: Using JSON encoding")
endif()

# Export ${sm_name} SM library
set(${sm_upper}_SM_LIBRARY ${sm_name}_sm PARENT_SCOPE)
EOF
    
    log_success "Agent-side code generated in $agent_sm_dir"
}

# Generate dApp-side code
generate_dapp_code() {
    local sm_name="$1"
    local sm_upper="$(echo "$sm_name" | tr '[:lower:]' '[:upper:]')"
    local dapp_sm_dir="$DAPP_ROOT/src/${sm_name}"
    
    log_info "Generating dApp-side code for $sm_name SM..."
    
    # Create directory structure
    mkdir -p "$dapp_sm_dir/defs"
    
    # Copy ASN.1 file with standardized name
    local asn_filename="e3sm_${sm_name}.asn"
    cp "$ASN_FILE" "$dapp_sm_dir/defs/$asn_filename"
    
    # Generate __init__.py
    cat > "$dapp_sm_dir/__init__.py" << 'EOF'
import os
import sys

sys.path.append(os.path.realpath(os.getcwd()))
EOF

    # Generate main dApp implementation
    cat > "$dapp_sm_dir/${sm_name}_dapp.py" << 'EOFPYTHON'
#!/usr/bin/env python3
"""
${sm_name} Service Model dApp Implementation
Generated by E3 SM Generator
"""

__author__ = "Generated by E3 SM Generator - Feel free to put your name, but please keep the ack \"Initial skeleton based on the E3 Template Generator - Andrea Lacava\""

import time
import os
import asn1tools
from dapp.dapp import DApp
from e3interface.e3_logging import dapp_logger

class ${sm_name}DApp(DApp):
    """
    ${sm_name} Service Model dApp
    
    This dApp implements the ${sm_name} service model for E3.
    TODO: Add specific description of what this SM does.
    """
    
    def __init__(self, id: int = 1, link: str = 'posix', transport: str = 'ipc', 
                 encoding_method: str = "asn1", **kwargs):
        super().__init__(id=id, link=link, transport=transport, encoding_method=encoding_method, **kwargs)
        
        # Initialize ${sm_name} encoder based on encoding method
        self._init_${sm_name}_encoder()
        
        # Register callback for indication messages
        self.e3_interface.add_callback(self.dapp_id, self.process_indication_message)
        
        # TODO: Add SM-specific configuration parameters
        
        dapp_logger.info(f"${sm_name} dApp initialized with encoding method: {self.encoding_method}")
    
    def _init_${sm_name}_encoder(self):
        """Initialize the ${sm_name} encoder based on the encoding method"""
        match self.encoding_method:
            case "asn1":
                asn_file_path = os.path.join(os.path.dirname(__file__), "defs", "e3sm_${sm_name}.asn")
                self.${sm_name}_encoder = asn1tools.compile_files(asn_file_path, codec="per")
                dapp_logger.info(f"Loaded ASN.1 schema from {asn_file_path}")
            case "json":
                # Future: Initialize JSON encoder
                self.${sm_name}_encoder = None
                dapp_logger.error("JSON encoding not yet implemented")
                raise NotImplementedError("JSON encoding not yet implemented")
            case _:
                raise ValueError(f"Unsupported encoding method: {self.encoding_method}")
    
    def _encode_${sm_name}_message(self, message_type: str, data: dict) -> bytes:
        """Encode a ${sm_name} message using the configured encoding method
        
        Args:
            message_type: The ${sm_name} message type to encode
            data: The data dictionary to encode
            
        Returns:
            Encoded bytes
        """
        if self.encoding_method == "asn1":
            if self.${sm_name}_encoder is None:
                raise RuntimeError("ASN.1 encoder not initialized")
            return self.${sm_name}_encoder.encode(message_type, data)
        elif self.encoding_method == "json":
            # Future: Implement JSON encoding
            import json
            return json.dumps(data).encode('utf-8')
        else:
            raise ValueError(f"Unsupported encoding method: {self.encoding_method}")
    
    def _decode_${sm_name}_message(self, message_type: str, data: bytes) -> dict:
        """Decode a ${sm_name} message using the configured encoding method
        
        Args:
            message_type: The ${sm_name} message type to decode
            data: The encoded bytes to decode
            
        Returns:
            Decoded data dictionary
        """
        if self.encoding_method == "asn1":
            if self.${sm_name}_encoder is None:
                raise RuntimeError("ASN.1 encoder not initialized")
            return self.${sm_name}_encoder.decode(message_type, data)
        elif self.encoding_method == "json":
            # Future: Implement JSON decoding
            import json
            return json.loads(data.decode('utf-8'))
        else:
            raise ValueError(f"Unsupported encoding method: {self.encoding_method}")
    
    def process_indication_message(self, dapp_identifier, data):
        """Process indication messages from the agent
        
        Args:
            dapp_identifier: The dApp ID
            data: Encoded indication data
        """
        dapp_logger.debug(f'Triggered callback for dApp {dapp_identifier}')
        
        # TODO: Decode the indication message
        # Example:
        # indication_data = self._decode_${sm_name}_message("${sm_name^}-IndicationType", data)
        # dapp_logger.info(f"Received indication: {indication_data}")
        
        # TODO: Process the indication data
        # TODO: Generate control messages if needed
        
        pass
    
    def send_control_message(self, control_data: dict) -> bool:
        """Send a control message to the agent
        
        Args:
            control_data: Dictionary containing control data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # TODO: Encode the control message
            # Example:
            # encoded_control = self._encode_${sm_name}_message("${sm_name^}-ControlType", control_data)
            # self.e3_interface.schedule_control(dappId=self.dapp_id, actionData=encoded_control)
            
            dapp_logger.info("Control message sent successfully")
            return True
        except Exception as e:
            dapp_logger.error(f"Failed to send control message: {e}")
            return False
    
    def _control_loop(self):
        """Main control loop - implement SM-specific logic here"""
        # TODO: Implement the main control loop
        # This method is called repeatedly by the base DApp class
        
        # Example: Process queued data, update visualizations, etc.
        time.sleep(0.1)  # Prevent busy waiting
    
    def _stop(self):
        """Cleanup when stopping the dApp"""
        # TODO: Add SM-specific cleanup
        dapp_logger.info("${sm_name} dApp stopped")
EOFPYTHON

    log_success "dApp-side code generated in $dapp_sm_dir"
}

# Update SM registry to include new SM
update_sm_registry() {
    local sm_name="$1"
    local sm_upper="$(echo "$sm_name" | tr '[:lower:]' '[:upper:]')"
    local registry_file="$AGENT_ROOT/openair2/E3AP/service_models/sm_registry.c"
    
    log_info "Updating SM registry to include $sm_name SM..."
    
    # Add extern declaration
    if ! grep -q "extern e3_service_model_t ${sm_name}_sm;" "$registry_file"; then
        sed -i "/extern e3_service_model_t spectrum_sm;/a extern e3_service_model_t ${sm_name}_sm;" "$registry_file"
    fi
    
    # Add registration call
    if ! grep -q "sm_registry_register(&${sm_name}_sm)" "$registry_file"; then
        sed -i "/sm_registry_register(&spectrum_sm);/a \\    ret = sm_registry_register(&${sm_name}_sm);\\
    if (ret != SM_SUCCESS) {\\
        LOG_E(E3AP, \"Failed to register ${sm_name} SM: %d\\\\n\", ret);\\
        return ret;\\
    }" "$registry_file"
    fi
    
    log_success "SM registry updated"
}

# Update service models CMakeLists.txt
update_service_models_cmake() {
    local sm_name="$1"
    local sm_upper="$(echo "$sm_name" | tr '[:lower:]' '[:upper:]')"
    local cmake_file="$AGENT_ROOT/openair2/E3AP/service_models/CMakeLists.txt"
    
    log_info "Updating service models CMakeLists.txt..."
    
    # Add subdirectory
    if ! grep -q "add_subdirectory(${sm_name}_sm)" "$cmake_file"; then
        sed -i "/add_subdirectory(spectrum_sm)/a add_subdirectory(${sm_name}_sm)" "$cmake_file"
    fi
    
    # Add library link
    if ! grep -q "\${${sm_upper}_SM_LIBRARY}" "$cmake_file"; then
        sed -i "/\${SPECTRUM_SM_LIBRARY}/a \\    \${${sm_upper}_SM_LIBRARY}" "$cmake_file"
    fi
    
    log_success "Service models CMakeLists.txt updated"
}

# Main function
main() {
    echo -e "${GREEN}E3 Service Model Generator${NC}"
    echo "=========================="
    echo ""
    
    # Parse arguments
    parse_args "$@"
    
    # Interactive mode
    if [[ "$INTERACTIVE" == true ]]; then
        prompt_interactive
    fi
    
    # Validate inputs
    validate_inputs
    
    # Generate code based on target
    if [[ "$TARGET" == "agent" || "$TARGET" == "both" ]]; then
        generate_agent_code "$SM_NAME"
        update_sm_registry "$SM_NAME"
        update_service_models_cmake "$SM_NAME"
    fi
    
    if [[ "$TARGET" == "dapp" || "$TARGET" == "both" ]]; then
        if [[ -d "$DAPP_ROOT" ]]; then
            generate_dapp_code "$SM_NAME"
        else
            log_warning "dApp root directory not found: $DAPP_ROOT"
            log_warning "Skipping dApp code generation"
        fi
    fi
    
    echo ""
    log_success "E3 Service Model generation completed!"
    echo ""
    echo "Next steps:"
    echo "1. Review and customize the generated code"
    echo "2. Implement SM-specific logic in TODO sections"
    echo "3. Add ASN.1-specific structures to CMakeLists.txt"
    echo "4. Build and test the service model"
    echo ""
    echo "Generated files:"
    if [[ "$TARGET" == "agent" || "$TARGET" == "both" ]]; then
        echo "  Agent: $AGENT_ROOT/openair2/E3AP/service_models/${SM_NAME}_sm/"
    fi
    if [[ "$TARGET" == "dapp" || "$TARGET" == "both" ]]; then
        echo "  dApp:  $DAPP_ROOT/src/${SM_NAME}/"
    fi
}

# Run main function
main "$@"