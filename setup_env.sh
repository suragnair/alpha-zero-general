DL_ENV=$1
NOTEBOOK_PORT=${2:-8888}
VISDOM_PORT=${3:-8097}
PYTORCH_IMAGE=pytorch:0.4.0-py3-gpu

if [ "$DL_ENV"=="pytorch" ]; then
    if [[ ! $(docker images -q ${pytorch:0.4.0-py3-gpu}) ]]; then
           docker build . -t ${PYTORCH_IMAGE} -f ./docker/Dockerfile.pytorch
    fi
    # this should run a pytorch notebook container
    docker run --shm-size 8G -v `pwd`:/workspace -p ${NOTEBOOK_PORT}:8888 -p ${VISDOM_PORT}:8097 --name pytorch_notebook ${PYTORCH_IMAGE}
    docker exec pytorch_notebook jupyter notebook list
else
    exit 1
fi