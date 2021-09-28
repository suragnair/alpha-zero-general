#!/bin/bash

DL_ENV=${1:-"pytorch"}
NOTEBOOK_PORT=${2:-8888}
VISDOM_PORT=${3:-8097}
PYTORCH_IMAGE=pytorch:0.4.1-py3-gpu

if [ ${DL_ENV}=="pytorch" ]; then
    if [[ ! $(docker images -q ${PYTORCH_IMAGE}) ]]; then
           docker build . -t ${PYTORCH_IMAGE} -f ./docker/Dockerfile.pytorch
    fi
    # this should run a pytorch notebook container
    docker run --runtime=nvidia --shm-size 8G -v `pwd`:/workspace -p ${NOTEBOOK_PORT}:8888 -p ${VISDOM_PORT}:8097 --name pytorch_notebook ${PYTORCH_IMAGE}
    docker exec pytorch_notebook jupyter notebook list
else
    exit 1
fi
