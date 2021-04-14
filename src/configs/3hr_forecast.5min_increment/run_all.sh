
#!/bin/bash

cd /data1/mloc/src/configs/3hr_forecast.5min_increment/


echo 'STARTING temperature'

./preprocess.temperature.sh
./extract_features.temperature.sh

echo 'DONE'

echo 'STARTING wind_speed'

./preprocess.wind_speed.sh
./extract_features.wind_speed.sh

echo 'DONE'

echo 'STARTING relative_humidity'

./preprocess.relative_humidity.sh
./extract_features.relative_humidity.sh

echo 'DONE'

echo 'STARTING pressure'

./preprocess.pressure.sh
./extract_features.pressure.sh

echo 'DONE'

echo 'STARTING log_Cn2_10T'

./preprocess.log_Cn2_10T.sh
./extract_features.log_Cn2_10T.sh

echo 'DONE'
