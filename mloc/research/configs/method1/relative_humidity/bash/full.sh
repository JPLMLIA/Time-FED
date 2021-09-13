#!/bin/bash

conda activate mloc

mkdir /data1/mloc/local/runs/method1
mkdir /data1/mloc/local/runs/method1/relative_humidity
mkdir /data1/mloc/local/runs/method1/relative_humidity/full

cd /data1/mloc/research/

python preprocess.py       -c configs/method1/relative_humidity/full.yml -s preprocess
python extract_features.py -c configs/method1/relative_humidity/full.yml -s extract_features
python classify.py         -c configs/method1/relative_humidity/full.yml -s classify

echo 'DONE'
