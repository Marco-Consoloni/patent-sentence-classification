#!/bin/sh

docker build . -t patent-sentence-classification:latest 

docker run \
    -it \
    --name patent-sentence-classification \
    --rm \
    --runtime=nvidia \
    --gpus all \
    --shm-size=1000gb \
    -v ./data:/app/data \
    -v ./models:/app/models \
    -v ./results:/app/results\
    patent-sentence-classification:latest \
    test.py