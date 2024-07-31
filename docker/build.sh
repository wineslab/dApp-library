#!/bin/bash

# This file can be use to perform a build standalone of the ns-3 container

docker build --build-arg token=fake -t ns-o-ran-online-env -f Dockerfile .
docker stop ns-o-ran-online-env
docker rm ns-o-ran-online-env
docker run -d -it --name ns-o-ran-online-env ns-o-ran-online-env
docker exec -it ns-o-ran-online-env bash