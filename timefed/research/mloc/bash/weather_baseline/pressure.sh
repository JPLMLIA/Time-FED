#!/bin/bash

mkdir /data1/mloc/local/runs/weather_baseline
mkdir /data1/mloc/local/runs/weather_baseline/pressure

cd /data1/mloc/src/

python preprocess.py       -c configs/weather_baseline/pressure.yml -s preprocess
python extract_features.py -c configs/weather_baseline/pressure.yml -s extract_features
python preprocess.py       -c configs/weather_baseline/pressure.yml -s preprocess_features
python classify.py         -c configs/weather_baseline/pressure.yml -s classify

echo 'DONE'
