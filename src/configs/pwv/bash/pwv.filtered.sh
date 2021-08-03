#!/bin/bash

conda activate mloc

mkdir /data1/mloc/local/runs/vapor
mkdir /data1/mloc/local/runs/vapor/filtered

cd /data1/mloc/src/

python preprocess.py       -c configs/pwv/pwv.filtered.yml -s preprocess
python extract_features.py -c configs/pwv/pwv.filtered.yml -s extract_features
python classify.py         -c configs/pwv/pwv.filtered.yml -s classify

echo 'DONE'
