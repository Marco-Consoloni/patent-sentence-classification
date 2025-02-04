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
    -e WANDB_API_KEY=bd21ffc2680605fb361dbc7bb99f651fe2f4d187 \
    patent-sentence-classification:latest \
    train.py "$@"