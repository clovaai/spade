#!/bin/bash
config_file=cord.1.5layers.test.yaml
file=main.py
cuda_n=1
arg1=" $config_file -m test "
export CUDA_VISIBLE_DEVICES=$cuda_n
python3 $file $arg1
