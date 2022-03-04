#!/bin/bash

conda activate mloc

cd /data1/mloc/src/bash/r0_baseline.5mins.historical/

./5ptwin.5min_shift.sh
./5ptwin.1hour_shift.sh
./5ptwin.2hour_shift.sh
./5ptwin.3hour_shift.sh

./10ptwin.5min_shift.sh
./10ptwin.1hour_shift.sh
./10ptwin.2hour_shift.sh
./10ptwin.3hour_shift.sh

echo 'DONE'
