#!/bin/bash

script_dir="$( dirname -- "$0"; )";

python3 "${script_dir}"/custom/CIFAR10_ResNet_Trainer.py
python3 "${script_dir}"/custom/CIFAR10_ResNet_Tester.py
