echo "weather"
python data2df.py -i ../../data/weather -o weather.h5 -d 'weather'
echo "bls"
python data2df.py -i ../../data/bls -o bls.h5 -d 'bls'
echo "r0 day"
python data2df.py -i ../../data/r0/v1.0.r0_day.mat -o r0_day.h5 -d 'r0-day'
echo "r0 night"
python data2df.py -i ../../data/r0/v2.0.r0_night.mat -o r0_night.h5 -d 'r0-night'
