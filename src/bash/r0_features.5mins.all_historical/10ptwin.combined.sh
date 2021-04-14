


#!/bin/bash

conda activate mloc

mkdir /data1/mloc/local/runs/r0_features.5mins.all_historical
mkdir /data1/mloc/local/runs/r0_features.5mins.all_historical/10ptwin.combined

cd /data1/mloc/src/

exec 2>&1 | tee /data1/mloc/local/runs/r0_features.5mins.all_historical/10ptwin.combined/run.log

python preprocess.py       -c configs/r0_features.5mins.all_historical/10ptwin.combined.yml -s preprocess
python extract_features.py -c configs/r0_features.5mins.all_historical/10ptwin.combined.yml -s extract_features
python classify.py         -c configs/r0_features.5mins.all_historical/10ptwin.combined.yml -s classify-1
python classify.py         -c configs/r0_features.5mins.all_historical/10ptwin.combined.yml -s classify-2
python classify.py         -c configs/r0_features.5mins.all_historical/10ptwin.combined.yml -s classify-3

echo 'DONE'
