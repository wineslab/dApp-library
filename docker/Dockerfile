#==================================================================================
#       Copyright (c) 2024 Northeastern University
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#==================================================================================
FROM ubuntu:22.04

# Set timezone
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install required packages
RUN apt-get update && apt-get install -y htop build-essential cmake automake libtool bison \
    flex git python3 python3-dev swig libsctp-dev

# Set the working directory
WORKDIR /workspace

# Clone the asn1c repository and build it
RUN git clone https://github.com/nokia/asn1c 
WORKDIR /workspace/asn1c 

RUN test -f configure || autoreconf -iv
RUN ./configure
RUN make
RUN make install

# Set the working directory for building the project
WORKDIR /workspace

# Copy the ASN.1 file, SWIG interface file, and C client source code into the container
COPY e3.asn /workspace/
COPY e3.i /workspace/
COPY client.c /workspace/

# Create an output directory for the compiled files
RUN mkdir -p /workspace/output

# Compile the ASN.1 files
RUN mkdir -p /workspace/output/asn1c_output && \
    asn1c -fcompound-names -gen-PER -no-gen-OER -no-gen-example -D /workspace/output/asn1c_output e3.asn

# Compile the C code for ASN.1
RUN gcc -c -o /workspace/output/asn1c_output/*.o /workspace/output/asn1c_output/*.c

# Generate the SWIG wrapper
RUN swig -python -o /workspace/output/e3_wrap.c e3.i

# Compile the SWIG wrapper and ASN.1 C code into a shared object
RUN gcc -shared -o /workspace/output/_e3.so /workspace/output/e3_wrap.c /workspace/output/asn1c_output/*.o -I/usr/include/python3.8

# Compile the C client
RUN gcc -o /workspace/output/client client.c /workspace/output/asn1c_output/*.o -lsctp

CMD ["/bin/bash"]
