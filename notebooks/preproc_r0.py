import sys
import os
import pandas as pd
sys.path.append('../src')
import datetime
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from plot_utils import *

# constants

F_INPUT = '../../data/v2/data.h5'
ds_label_all = 'r0' # ds stands for data source
ds_label_day = 'r0_day'
ds_label_ngt = 'r0_night'

DENSE_SUBSET_START = '2018-05-03'
DENSE_SUBSET_END = '2020-12-30'

smooth_mins = [1, 2, 5, 10, 15, 20]
r0_map = {'all':'r0_all','day':'r0_day','ngt':'r0_ngt'} # adn = all, day, ngt

feats_mnus_cn2 = ['pressure', 'relative_humidity', 'temperature', 'wind_speed', 'solar_zenith_angle','dayofyear', 'hour']
feats_plus_cn2 = feats_mnus_cn2 + ['logCn2']
feats_map = {'feats_plus_cn2': feats_plus_cn2, 'feats_mnus_cn2': feats_mnus_cn2}

# read data

df = pd.read_hdf(F_INPUT, 'merged')

# add new features

df['dayofyear'] = df.index.dayofyear
df['hour'] = df.index.hour
df['logCn2'] = np.log10(df['Cn2'])

# apply smoothing

for m in smooth_mins:
    for ds_label in [ds_label_all, ds_label_day, ds_label_ngt]:
        if ds_label == 'r0':
            label = ds_label + '_all_{}T'.format(m)
        elif ds_label == 'r0_night':
            label = 'r0_ngt_{}T'.format(m)
        else:
            label = ds_label + '_{}T'.format(m)
        if m == 1:
            df[label] = df[ds_label]
        else:
            df[label] = df[ds_label].rolling('{}T'.format(m)).mean()

# restricting data to usable, relatively dense subset

df_subset = df[(df.index > DENSE_SUBSET_START) & (df.index < DENSE_SUBSET_END)]

# resampling back down to 5 mins

df_subset_resamp5 = df_subset.resample('5 min').median()
df_subset_resamp5.to_hdf('df_subset_resamp5.r0.h5', 'df_subset_resamp5')

# taking the train and test
 
split_date = '2019-12-31'
train = df_subset.index <= split_date
test  = df_subset.index > split_date

# creating non-nan masks

valid_masks = {}
for f in feats_map.keys(): 
    valid_masks[f] = {}
    for r in r0_map.keys():
        valid_masks[f][r] = {}
        for m in smooth_mins:
            label = '{}_{}T'.format(r0_map[r], str(m))
            feats_plus_label = feats_map[f] + [label]
            valid_masks[f][r][m] = ~df_subset[feats_plus_label].isnull().any(axis=1)
            print("{} {} {:2} {:7} {:7}".format(f, r, m, np.sum(train & valid_masks[f][r][m]), np.sum(test & valid_masks[f][r][m])))


with open('valid_masks.r0.pkl', 'wb') as fh:
    pickle.dump(valid_masks, fh)

# train and test subroutine

def train_and_test(train_df, test_df, feats, label):
    regr = RandomForestRegressor(n_estimators=100, random_state=0)
    forest = regr.fit(train_df[feats], train_df[label])
#     r2 = regr.score(test_df[feats], test_df[label])
    preds = regr.predict(test_df[feats])
    r2 = r2_score(test_df[label], preds)
    sq_err = mean_squared_error(test_df[label], preds)
    perc_err = mean_absolute_percentage_error(test_df[label], preds)
    return {'forest': forest, 'preds': preds, 'r2': r2, 'sq_err': sq_err, 'perc_err': perc_err}

# Get All Results

results = {}
for f in feats_map.keys(): 
    results[f] = {}
    for r in r0_map.keys():
        results[f][r] = {}
        for m in smooth_mins:
            print("Progress: {} {} {}".format(f, r, m))
            label = '{}_{}T'.format(r0_map[r], str(m))
            feats = feats_map[f]
            valid = valid_masks[f][r][m]
            results[f][r][m] = train_and_test(df_subset.loc[train & valid], df_subset.loc[test & valid], feats, label)
            with open('results.r0.pkl', 'wb') as fh:
                pickle.dump(results, fh)
