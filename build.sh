#!/bin/bash

cd env && docker build --build-arg NB_USER=$USER --build-arg NB_UID=$(id -u) --build-arg NB_GID=$(id -g) --build-arg NB_DISPLAY=$DISPLAY -t coinrun_adventure -f Dockerfile .