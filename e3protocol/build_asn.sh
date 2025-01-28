#!/bin/bash
set -e
# set -x

# Set up paths
ASN1_FILE="../src/e3interface/defs/e3.asn"
ASN1C_OUTPUT_DIR="asn1c_output"
ASN1C_COMPILED_DIR="asn1c_compiled_output"

echo "Compiling ASN.1 file with asn1c..."
mkdir -p $ASN1C_OUTPUT_DIR
asn1c -pdu=auto -fno-include-deps -fcompound-names -findirect-choice -gen-PER -gen-OER -no-gen-example -D $ASN1C_OUTPUT_DIR $ASN1_FILE

# This is necessary to build correctly the OER
cp /opt/asn1c/share/asn1c/BIT_STRING_oer.c ./$ASN1C_OUTPUT_DIR

echo "Compiling ASN.1 generated C code..."
ABS_PATH="$(pwd)"
mkdir -p $ASN1C_COMPILED_DIR
cd $ASN1C_COMPILED_DIR
gcc -c -fPIC $ABS_PATH/$ASN1C_OUTPUT_DIR/*.c -I $ABS_PATH/$ASN1C_OUTPUT_DIR
cd -

echo "Compiling C client..." 
gcc -o client/client client/client.c wrapper/e3.c $ASN1C_COMPILED_DIR/*.o -I$ASN1C_OUTPUT_DIR -Idefs/ -lsctp -ggdb

echo "Build completed."
