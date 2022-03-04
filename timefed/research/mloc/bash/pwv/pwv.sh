#!/bin/bash

mkdir /data1/urebbapr/MLOC/local/runs/pwv

cd /data1/urebbapr/MLOC/src/

F_CONFIG=configs/pwv/pwv.3hr_forecast.30min_increment.tsfresh.yml

python preprocess.py       -c $F_CONFIG -s preprocess
#python extract_features.py -c $F_CONFIG -s extract_features
#python classify.py         -c $F_CONFIG -s classify

echo 'DONE'
