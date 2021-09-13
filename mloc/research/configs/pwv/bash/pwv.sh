#!/bin/bash

conda activate mloc

mkdir /data1/mloc/local/runs/vapor/

cd /data1/mloc/research/

python preprocess.py       -c configs/pwv/pwv.yml -s preprocess
python extract_features.py -c configs/pwv/pwv.yml -s extract_features
python classify.py         -c configs/pwv/pwv.yml -s classify

echo 'DONE'
