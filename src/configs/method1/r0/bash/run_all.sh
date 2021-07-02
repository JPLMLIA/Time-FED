#!/bin/bash

cd /data1/mloc/src/configs/method1/r0/bash/

#./r0.cn2.weather.historical.sh
./r0.cn2.weather.sh -se
./r0.weather.historical.sh
./r0.weather.sh

echo 'FINISHED ALL'
