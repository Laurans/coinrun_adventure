#!/bin/bash

cd env && poetry export -f requirements.txt > requirements.txt && docker build --build-arg NB_USER=$USER --build-arg NB_UID=$(id -u) --build-arg NB_GID=$(id -g) -t coinrun_adventure -f Dockerfile .