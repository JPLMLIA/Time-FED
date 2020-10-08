"""
Various utility functions
"""
import numpy  as np
import pandas as pd

from glob import glob
from tqdm import tqdm

#%%
mappings = {
    'H_PAAVG1M': 'pressure',
    'H_RHAVG1M': 'relative_humidity',
    'H_TAAVG1M': 'temperature',
    'H_WDAVG2M': 'wind_direction',
    'H_WSAVG2M': 'wind_speed',
}

def load_weather(path):
    print('Loading in weather data')
    df = None
    for code, column in mappings.items():
        files = sorted(glob(f'{path}/{code}/**/*'))
        _df = pd.DataFrame()
        for file in tqdm(files, desc=f'Compiling {column}'):
            _df = pd.concat([
                _df,
                pd.read_csv(file, sep='\t', header=None, names=['ts', column], index_col='ts', parse_dates=True, dtype={column: float}, na_values='///')
            ])
        else:
            if df is not None:
                df = df.join(_df, sort=True)
            else:
                df = _df

    print('Done')
    return df
