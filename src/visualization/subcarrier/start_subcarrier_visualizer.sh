#!/bin/bash

#
# SPDX-FileCopyrightText: Copyright (c) 2026 Northeastern University. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WEB_PORT=5001        # HTTP port  (0 disables HTTP)
HTTPS_PORT=0         # HTTPS port (0 = off; HTTPS is normally provided by the OpenShift route).
                     # Set >0 only to serve TLS directly from the app (then pass --cert/--key,
                     # e.g. an OpenShift service-serving cert mount; otherwise it self-signs).
ZMQ_PORT=5559
CERT=""              # optional TLS cert PEM (used only when --https-port > 0)
KEY=""               # optional TLS key  PEM (used only when --https-port > 0)

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)       WEB_PORT="$2";   shift 2 ;;
        --https-port) HTTPS_PORT="$2"; shift 2 ;;
        --cert)       CERT="$2";       shift 2 ;;
        --key)        KEY="$2";        shift 2 ;;
        --zmq-port)   ZMQ_PORT="$2";   shift 2 ;;
        *) echo "Unknown option: $1";
           echo "Usage: $0 [--port HTTP_PORT] [--https-port HTTPS_PORT] [--cert PEM] [--key PEM] [--zmq-port ZMQ_PORT]";
           echo "       (--port 0 disables HTTP, --https-port 0 disables HTTPS)";
           exit 1 ;;
    esac
done

echo "=========================================="
echo "Subcarrier Power Visualizer"
echo "=========================================="
echo "HTTP Port:  $WEB_PORT"
echo "HTTPS Port: $HTTPS_PORT"
echo "ZMQ Port:   $ZMQ_PORT"
echo "=========================================="

python3 -c "import flask, flask_sock" 2>/dev/null || {
    echo "Error: flask / flask-sock not installed."
    echo "Install with: pip3 install -r applications/subcarrier-power/deps/requirements.txt"
    exit 1
}

ARGS=(--port "$WEB_PORT" --https-port "$HTTPS_PORT" --zmq-port "$ZMQ_PORT")
[[ -n "$CERT" ]] && ARGS+=(--cert "$CERT")
[[ -n "$KEY"  ]] && ARGS+=(--key  "$KEY")

python3 subcarrier_visualizer.py "${ARGS[@]}"
