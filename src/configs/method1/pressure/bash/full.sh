#!/bin/bash

conda activate mloc

mkdir /data1/mloc/local/runs/method1
mkdir /data1/mloc/local/runs/method1/pressure
mkdir /data1/mloc/local/runs/method1/pressure/full

cd /data1/mloc/src/

python preprocess.py       -c configs/method1/pressure/full.yml -s preprocess
python extract_features.py -c configs/method1/pressure/full.yml -s extract_features
python classify.py         -c configs/method1/pressure/full.yml -s classify

echo 'DONE'
