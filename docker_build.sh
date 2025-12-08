#!/bin/bash

CONTAINER_NAME="aLlama"

docker exec -it $CONTAINER_NAME bash -c "cd workspace/lcdev && bash ./build_test.sh"
