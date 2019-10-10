#!/bin/bash

image_name="coinrun_adventure"
container_name="coinrun_venv"
if [ ! "$(docker ps -q -f name=$container_name)" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=$container_name)" ]; then
        if [ "$1" == "rm" ]; then
            docker rm $container_name
        else
            docker start -i $container_name
        fi

    else
        docker run --gpus all -it -v $PWD:/home/vscode/workdir --name $container_name $image_name
    fi
fi 