#!/bin/bash

conda activate mloc

mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment
mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment/cn2_weather

cd /data1/mloc/src/

exec 2>&1 | tee /data1/mloc/local/runs/3hr_forecast.5min_increment/cn2_weather/run.log

# python preprocess.py       -c configs/3hr_forecast.5min_increment/cn2_weather.yml -s preprocess
python extract_features.py -c configs/3hr_forecast.5min_increment/cn2_weather.yml -s extract_features -se
python classify.py         -c configs/3hr_forecast.5min_increment/cn2_weather.yml -s classify

echo 'DONE'
