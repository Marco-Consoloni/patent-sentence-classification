#!/bin/sh

#sdocker build . -t patent-sentence-classification:latest 

docker run \
    -it \
    --name patent-sentence-classification \
    --rm \
    --runtime=nvidia \
    --gpus all \
    --shm-size=1000gb \
    -v ./data:/app/data \
    -v ./models:/app/models \
    -v ./results/finetuning/:/app/results/finetuning/ \
    -v ./patents:/app/patents \
    -v ./test.py:/app/test.py \
    patent-sentence-classification:latest \
    test.py "$@"