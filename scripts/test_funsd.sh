#!/bin/bash
config_file=funsd.1.5layers.test.yaml
dsetL=(ocr-nfs3)
file=main.py
cuda_n=6
arg1=" $config_file -m test "
export CUDA_VISIBLE_DEVICES=$cuda_n
python3 $file $arg1