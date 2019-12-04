#!/bin/bash

pip install --upgrade pip
pip install --upgrade setuptools
poetry install
cp .netrc ../
