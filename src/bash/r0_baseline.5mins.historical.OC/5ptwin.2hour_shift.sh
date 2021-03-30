#!/bin/bash

conda activate mloc

mkdir /data1/mloc/local/runs/r0_baseline.5mins.historical
mkdir /data1/mloc/local/runs/r0_baseline.5mins.historical/5ptwin.2hour_shift

cd /data1/mloc/src/

python preprocess.py       -c configs/r0_baseline.5mins.historical/5ptwin.2hour_shift.yml -s preprocess
python extract_features.py -c configs/r0_baseline.5mins.historical/5ptwin.2hour_shift.yml -s extract_features
python preprocess.py       -c configs/r0_baseline.5mins.historical/5ptwin.2hour_shift.yml -s preprocess_features
python classify.py         -c configs/r0_baseline.5mins.historical/5ptwin.2hour_shift.yml -s classify

echo 'DONE'
