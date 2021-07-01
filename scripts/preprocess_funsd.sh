#!/bin/bash
wget https://guillaumejaume.github.io/FUNSD/dataset.zip
mkdir -p data/funsd/original
mv dataset.zip data/funsd/original
unzip data/funsd/original/dataset.zip -d data/funsd/original/

config_file=funsd.preprocess.yaml
python main.py $config_file -m preprocess