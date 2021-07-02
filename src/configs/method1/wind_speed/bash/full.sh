#!/bin/bash

conda activate mloc

mkdir /data1/mloc/local/runs/method1
mkdir /data1/mloc/local/runs/method1/wind_speed
mkdir /data1/mloc/local/runs/method1/wind_speed/full

cd /data1/mloc/src/

python preprocess.py       -c configs/method1/wind_speed/full.yml -s preprocess
python extract_features.py -c configs/method1/wind_speed/full.yml -s extract_features
python classify.py         -c configs/method1/wind_speed/full.yml -s classify

echo 'DONE'
