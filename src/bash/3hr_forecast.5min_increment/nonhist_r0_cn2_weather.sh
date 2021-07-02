#!/bin/bash

conda activate mloc

mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment
mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment/nonhist_r0_cn2_weather

cd /data1/mloc/src/

python preprocess.py       -c configs/3hr_forecast.5min_increment/nonhist_r0_cn2_weather.yml -s preprocess
python extract_features.py -c configs/3hr_forecast.5min_increment/nonhist_r0_cn2_weather.yml -s extract_features
python classify.py         -c configs/3hr_forecast.5min_increment/nonhist_r0_cn2_weather.yml -s classify

echo 'DONE'
