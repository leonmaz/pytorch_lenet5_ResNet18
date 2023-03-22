#!/usr/bin/env bash

export projectname='fed_project_env'
export projectpath="."

python3 -m venv ${projectname}
source ${projectname}/bin/activate

pip install --upgrade pip
pip install -r ./virtualenv/min-requirements.txt
