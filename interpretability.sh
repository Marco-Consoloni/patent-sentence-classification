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
    -v ./data/patents:/workspace/data/patents \
    -v ./models/finetuning:/workspace/models/finetuning \
    -v ./models/incremental:/workspace/models/incremental \
    -v ./results:/workspace/results\
    -v ./src:/workspace/src \
    -v ./config.yaml://workspace/config.yaml \
    interpretability