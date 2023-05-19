#!/bin/bash

script_dir="$( dirname -- "$0"; )";

python3 "${script_dir}"/custom/CIFAR100_ResNet_Trainer.py
python3 "${script_dir}"/custom/CIFAR100_ResNet_Tester.py
