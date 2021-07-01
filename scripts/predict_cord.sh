#!/bin/bash
config_file=cord.1.5layers.test.yaml
file=main.py
#arg1=" $config_file -m predict -p ./data/samples/cord_predict.json "
arg1=" $config_file -m predict -p ./data/samples/cord_plus_plus_predict.json "

export CUDA_VISIBLE_DEVICES=6
python3 $file $arg1