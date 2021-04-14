
#!/bin/bash

cd /data1/mloc/src/configs/3hr_forecast.5min_increment/


echo 'STARTING temperature'

./preprocess.temperature.sh
./feature_extraction.temperature.sh

echo 'DONE'

echo 'STARTING wind_speed'

./preprocess.wind_speed.sh
./feature_extraction.wind_speed.sh

echo 'DONE'

echo 'STARTING relative_humidity'

./preprocess.relative_humidity.sh
./feature_extraction.relative_humidity.sh

echo 'DONE'

echo 'STARTING pressure'

./preprocess.pressure.sh
./feature_extraction.pressure.sh

echo 'DONE'

echo 'STARTING log_Cn2_10T'

./preprocess.log_Cn2_10T.sh
./feature_extraction.log_Cn2_10T.sh

echo 'DONE'
