#!/bin/bash

conda activate mloc

mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment
mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment/weather

cd /data1/mloc/src/

# Temperature
mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment/weather/temperature
mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment/weather/temperature/H5
python plots.py -c configs/3hr_forecast.5min_increment/plot.weather.yml \
                -s temperature-5 \
                -ki features/temperature/historical_5_min/test \
                -ko forecasts/temperature/H5 \
                -m /data1/mloc/local/runs/3hr_forecast.5min_increment/weather/temperature_H5_min.pkl

mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment/weather/temperature/H180
python plots.py -c configs/3hr_forecast.5min_increment/plot.weather.yml \
                -s temperature-180 \
                -ki features/temperature/historical_180_min/test \
                -ko forecasts/temperature/H180 \
                -m /data1/mloc/local/runs/3hr_forecast.5min_increment/weather/temperature_H180_min.pkl

# Pressure
mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment/weather/pressure
mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment/weather/pressure/H5
python plots.py -c configs/3hr_forecast.5min_increment/plot.weather.yml \
                -s pressure-5 \
                -ki features/pressure/historical_5_min/test \
                -ko forecasts/pressure/H5 \
                -m /data1/mloc/local/runs/3hr_forecast.5min_increment/weather/pressure_H5_min.pkl

mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment/weather/pressure/H180
python plots.py -c configs/3hr_forecast.5min_increment/plot.weather.yml \
                -s pressure-180 \
                -ki features/pressure/historical_180_min/test \
                -ko forecasts/pressure/H180 \
                -m /data1/mloc/local/runs/3hr_forecast.5min_increment/weather/pressure_H180_min.pkl

# relative_humidity
mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment/weather/relative_humidity
mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment/weather/relative_humidity/H5
python plots.py -c configs/3hr_forecast.5min_increment/plot.weather.yml \
                -s relative_humidity-5 \
                -ki features/relative_humidity/historical_5_min/test \
                -ko forecasts/relative_humidity/H5 \
                -m /data1/mloc/local/runs/3hr_forecast.5min_increment/weather/relative_humidity_H5_min.pkl

mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment/weather/relative_humidity/H180
python plots.py -c configs/3hr_forecast.5min_increment/plot.weather.yml \
                -s relative_humidity-180 \
                -ki features/relative_humidity/historical_180_min/test \
                -ko forecasts/relative_humidity/H180 \
                -m /data1/mloc/local/runs/3hr_forecast.5min_increment/weather/relative_humidity_H180_min.pkl

# wind_speed
mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment/weather/wind_speed
mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment/weather/wind_speed/H5
python plots.py -c configs/3hr_forecast.5min_increment/plot.weather.yml \
                -s wind_speed-5 \
                -ki features/wind_speed/historical_5_min/test \
                -ko forecasts/wind_speed/H5 \
                -m /data1/mloc/local/runs/3hr_forecast.5min_increment/weather/wind_speed_H5_min.pkl

mkdir /data1/mloc/local/runs/3hr_forecast.5min_increment/weather/wind_speed/H180
python plots.py -c configs/3hr_forecast.5min_increment/plot.weather.yml \
                -s wind_speed-180 \
                -ki features/wind_speed/historical_180_min/test \
                -ko forecasts/wind_speed/H180 \
                -m /data1/mloc/local/runs/3hr_forecast.5min_increment/weather/wind_speed_H180_min.pkl

echo 'DONE'
