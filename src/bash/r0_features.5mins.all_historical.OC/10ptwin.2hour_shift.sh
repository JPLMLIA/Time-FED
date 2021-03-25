#!/bin/bash

conda activate mloc

cd /data1/mloc/src/

python preprocess.py       -c configs/r0_features.5mins.all_historical/10ptwin.2hour_shift.yml -s preprocess
python extract_features.py -c configs/r0_features.5mins.all_historical/10ptwin.2hour_shift.yml -s extract_features
python preprocess.py       -c configs/r0_features.5mins.all_historical/10ptwin.2hour_shift.yml -s preprocess_features
python classify.py         -c configs/r0_features.5mins.all_historical/10ptwin.2hour_shift.yml -s classify

echo 'DONE'
