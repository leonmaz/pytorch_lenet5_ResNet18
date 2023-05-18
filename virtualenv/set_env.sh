#!/usr/bin/env bash

export projectname='fed_project_env'
export projectpath="."

virtualenv --python="python3.8" ${projectname} 

source ${projectname}/bin/activate

pip install --upgrade pip
pip install -r ./virtualenv/min-requirements.txt
