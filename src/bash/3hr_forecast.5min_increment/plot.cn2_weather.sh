#!/bin/bash

conda activate mloc

mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment
mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment/cn2_weather

cd /data1/mloc/src/

mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment/cn2_weather/H5
python plots.py -c configs/3hr_forecast.5min_increment/plot.cn2_weather.yml \
                -s classify-5 \
                -ki features/log_Cn2_10T/historical_5_min/test \
                -ko forecasts/log_Cn2_10T/H5 \
                -m /data1/mloc/local/runs/3hr_forecast.5min_increment/cn2_weather/log_Cn2_10T_H5_min.pkl

mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment/cn2_weather/H180
python plots.py -c configs/3hr_forecast.5min_increment/plot.cn2_weather.yml \
                -s classify-180 \
                -ki features/log_Cn2_10T/historical_180_min/test \
                -ko forecasts/log_Cn2_10T/H180 \
                -m /data1/mloc/local/runs/3hr_forecast.5min_increment/cn2_weather/log_Cn2_10T_H180_min.pkl

echo 'DONE'
