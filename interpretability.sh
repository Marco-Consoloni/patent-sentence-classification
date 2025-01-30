#!/bin/sh
#docker build . -t interpretability

docker run \
    -it \
    --name interpretability \
    --rm \
    --runtime=nvidia \
    --gpus all \
    --shm-size=1000gb \
    -p 8888:8888 \
    -v ./data:/workspace/data \
    -v ./models:/workspace/models \
    -v ./results:/workspace/results\
    -v ./patents:/workspace/patents \
    -v ./src:/workspace/src \
    -v ./config.yaml://workspace/config.yaml \
    interpretability