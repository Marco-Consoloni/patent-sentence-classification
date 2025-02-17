#!/bin/sh

#docker build . -t llama-prompting

docker run \
    -it \
    --name llama-prompting \
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
    -v ./config.yaml:/workspace/config.yaml \
    -v ./prompting/prompt_components.json:/workspace/prompting/prompt_components.json \
    -v ./prompting/prompt_templates.json:/workspace/prompting/prompt_templates.json \
    -v /vast/marco/patent-sentence-classification/huggingface_cache:/root/.cache/huggingface \
    llama-prompting

#-v ./models/huggingface:/root/.cache/huggingface \