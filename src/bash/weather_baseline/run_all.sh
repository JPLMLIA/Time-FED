#!/bin/bash

cd /data1/mloc/src/bash/weather_baseline/

./temperature.sh
./pressure.sh
./relative_humidity.sh
./wind_speed.sh

echo 'DONE'
