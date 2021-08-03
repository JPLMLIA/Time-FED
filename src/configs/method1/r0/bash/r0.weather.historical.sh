#!/bin/bash

conda activate mloc

mkdir /data1/mloc/local/runs/method1
mkdir /data1/mloc/local/runs/method1/r0
mkdir /data1/mloc/local/runs/method1/r0/r0.weather.historical

cd /data1/mloc/src/

python preprocess.py       -c configs/method1/r0/r0.weather.historical.yml -s preprocess
python extract_features.py -c configs/method1/r0/r0.weather.historical.yml -s extract_features
python classify.py         -c configs/method1/r0/r0.weather.historical.yml -s classify

echo 'DONE'
