#!/bin/sh

#docker build . -t patent-sentence-classification:latest
#docker build . --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t patent-sentence-classification:latest 

docker run \
    -it \
    --name patent-sentence-classification \
    --rm \
    --runtime=nvidia \
    --gpus all \
    --shm-size=1500gb \
    -v ./data:/app/data \
    -v ./data/incremental:/app/data/incremental \
    -v ./models:/app/models \
    -v /vast/marco/patent-sentence-classification/models/incremental:/app/models/incremental \
    -v ./results/finetuning/:/app/results/finetuning/ \
    -v ./results/incremental/:/app/results/incremental/ \
    -v ./train.py:/app/train.py \
    -v ./config.yaml:/app/config.yaml \
    -e WANDB_API_KEY=bd21ffc2680605fb361dbc7bb99f651fe2f4d187 \
    --user $(id -u):$(id -g) \
    patent-sentence-classification:latest \
    train.py "$@"


# -v ./models/incremental:/app/models/incremental \
