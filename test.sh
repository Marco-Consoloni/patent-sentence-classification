#!/bin/sh

#docker build . -t patent-sentence-classification:latest 
#docker build . --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t patent-sentence-classification:latest 

docker run \
    -it \
    --name patent-sentence-classification \
    --rm \
    --runtime=nvidia \
    --gpus all \
    --shm-size=1000gb \
    -v ./data:/app/data \
    -v ./data/incremental:/app/data/incremental \
    -v ./data/patents:/app/data/patents \
    -v ./models:/app/models \
    -v /vast/marco/patent-sentence-classification/models/incremental:/app/models/incremental/ \
    -v ./results/finetuning/:/app/results/finetuning/ \
    -v ./results/incremental/:/app/results/incremental/ \
    -v ./results/patents/:/app/results/patents \
    -v ./patents:/app/patents \
    -v ./test.py:/app/test.py \
    -v ./config.yaml:/app/config.yaml \
    --user $(id -u):$(id -g) \
    patent-sentence-classification:latest \
    test.py "$@"


# -v ./models/incremental:/app/models/incremental \
