#!/bin/bash

conda activate mloc

mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment
mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment/r0_cn2_weather

cd /data1/mloc/src/

mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment/r0_cn2_weather/H5
python plots.py -c configs/3hr_forecast.5min_increment/plot.r0_cn2_weather.yml \
                -s classify-5 \
                -ki features/r0_10T/historical_5_min/test \
                -ko forecasts/r0_10T/H5 \
                -m /data1/mloc/local/runs/3hr_forecast.5min_increment/r0_cn2_weather/r0_10T_H5_min.pkl

mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment/r0_cn2_weather/H180
python plots.py -c configs/3hr_forecast.5min_increment/plot.r0_cn2_weather.yml \
                -s classify-180 \
                -ki features/r0_10T/historical_180_min/test \
                -ko forecasts/r0_10T/H180 \
                -m /data1/mloc/local/runs/3hr_forecast.5min_increment/r0_cn2_weather/r0_10T_H180_min.pkl

echo 'DONE'
