#!/usr/bin/env bash
# Regenerate the checked-in protobuf Python stubs (*_pb2.py) for the E3SM
# protobuf codec path. Run after editing any .proto under src/*/defs.
#
# Requires: protoc (protobuf-compiler). The generated *_pb2.py are committed so
# the wheel needs no protoc at build/run time.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

gen() {
    local dir="$1"; shift
    echo "protoc -> ${dir}"
    ( cd "${dir}" && protoc --python_out=. --proto_path=. "$@" )
}

gen "${ROOT}/src/spectrum/defs" e3sm_spectrum.proto e3sm_oai_l1_kpm.proto
gen "${ROOT}/src/simple/defs"   e3sm_simple.proto

echo "Done. Review and commit the regenerated *_pb2.py."
