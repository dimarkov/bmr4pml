#!/bin/bash
docker run --rm -it -u $(id -u):$(id -g) -v $HOME:$HOME -v $(pwd):$(pwd)\
 -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
 -e HOME=$HOME\
 -e PYTHONPATH=$HOME/bmr4pml/\
 -w $(pwd)\
 --runtime=nvidia --gpus all bmr $@
