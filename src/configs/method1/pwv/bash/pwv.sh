#!/bin/bash

mkdir /data1/mloc/local/runs/method1
mkdir /data1/mloc/local/runs/method1/pwv
mkdir /data1/mloc/local/runs/method1/pwv/full

cd /data1/mloc/src/

python preprocess.py       -c configs/method1/pwv/full.yml -s preprocess
python extract_features.py -c configs/method1/pwv/full.yml -s extract_features
python classify.py         -c configs/method1/pwv/full.yml -s classify

echo 'DONE'
