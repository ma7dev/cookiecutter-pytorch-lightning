#!/bin/bash
CONFIG_PATH="../config/experiments.yml"
SELECTED="main" #"tracking_bce_blstm"
GPUS="0"
python experiments.py --config_path=$CONFIG_PATH --selected=$SELECTED --gpus=$GPUS