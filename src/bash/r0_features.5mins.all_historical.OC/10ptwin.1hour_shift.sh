#!/bin/bash

conda activate mloc

mkdir /data1/mloc/local/runs/r0_features.5mins.all_historical
mkdir /data1/mloc/local/runs/r0_features.5mins.all_historical/10ptwin.1hour_shift

cd /data1/mloc/src/

python preprocess.py       -c configs/r0_features.5mins.all_historical/10ptwin.1hour_shift.yml -s preprocess
python extract_features.py -c configs/r0_features.5mins.all_historical/10ptwin.1hour_shift.yml -s extract_features
python preprocess.py       -c configs/r0_features.5mins.all_historical/10ptwin.1hour_shift.yml -s preprocess_features
python classify.py         -c configs/r0_features.5mins.all_historical/10ptwin.1hour_shift.yml -s classify

echo 'DONE'
