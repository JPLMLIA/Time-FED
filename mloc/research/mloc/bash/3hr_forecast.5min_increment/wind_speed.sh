#!/bin/bash

conda activate mloc

mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment
mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment/wind_speed

cd /data1/mloc/src/

exec 2>&1 | tee /data1/mloc/local/runs/3hr_forecast.5min_increment/wind_speed/run.log

python preprocess.py       -c configs/3hr_forecast.5min_increment/wind_speed.yml -s preprocess
python extract_features.py -c configs/3hr_forecast.5min_increment/wind_speed.yml -s extract_features
python classify.py         -c configs/3hr_forecast.5min_increment/wind_speed.yml -s classify

echo 'DONE'
