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

FROM ubuntu:22.04 as base


# Add you ghp_ generate token as coommand argument
ARG token=fake

# Common Instructions
RUN mkdir -p /workspace
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y wget \
                                        build-essential libc6-dev linux-libc-dev\
                                        git \
                                        cmake autoconf automake libtool bison flex \
                                        libboost-all-dev tmux python3 curl \
                                        ninja-build pipenv \ 
                                        htop nano swig
WORKDIR /workspace

RUN ldconfig

RUN git clone -b enviroment https://${token}@github.com/wineslab/ns-o-ran-network-gym /workspace/ns-o-ran-network-gym
WORKDIR /workspace/ns-o-ran-network-gym
RUN pip3 install -r requirements.txt

# Cleanup
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["/bin/bash"]
