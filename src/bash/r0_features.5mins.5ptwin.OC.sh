#!/bin/bash

conda activate mloc

cd /data1/mloc/src/

python preprocess.py       -c configs/r0_features.5mins.5ptwin.OC.yml -s preprocess
python extract_features.py -c configs/r0_features.5mins.5ptwin.OC.yml -s extract_features
python preprocess.py       -c configs/r0_features.5mins.5ptwin.OC.yml -s preprocess_features
python classify.py         -c configs/r0_features.5mins.5ptwin.OC.yml -s classify

echo 'DONE'
