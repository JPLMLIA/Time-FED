#!/bin/bash

cd /data1/mloc/src/configs/method1

./temperature/full.sh
./pressure/full.sh
./relative_humidity/full.sh
./wind_speed/full.sh

echo 'DONE ALL'
