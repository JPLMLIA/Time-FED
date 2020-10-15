"""
Various utility functions
"""
import h5py
import numpy  as np
import pandas as pd

from glob import glob
from tqdm import tqdm


mappings = {
    'H_PAAVG1M': 'pressure',
    'H_RHAVG1M': 'relative_humidity',
    'H_TAAVG1M': 'temperature',
    'H_WDAVG2M': 'wind_direction',
    'H_WSAVG2M': 'wind_speed',
}

def load_weather(path):
    """
    Loads in weather data from .txt files
    """
    print('Loading in weather data')
    df = None
    for code, column in tqdm(mappings.items(), desc='Parameters', position=0):
        files = sorted(glob(f'{path}/{code}/**/*'))
        _df = pd.DataFrame()
        for file in tqdm(files, desc=f'Compiling {column}', position=1):
            _df = pd.concat([
                _df,
                pd.read_csv(file, sep='\t', header=None, names=['ts', column], index_col='ts', parse_dates=True, dtype={column: float}, na_values='///')
            ])
        else:
            _df = _df.resample('1T').mean()
            if df is not None:
                df = df.join(_df, sort=True)
            else:
                df = _df

    print('Done')
    return df

def load_cn2(path):
    """
    Loads in Cn2 .mat files, support for mat5.0 and mat7.3 files only
    """
    def mat50(file):
        hold = {i: [] for i in range(len(columns))}
        data = loadmat(file)
        for pair in data['YearData']:
            for i, value in enumerate(pair):
                hold[i].append(value)
        for i, arr in hold.items():
            tmp[columns[i]] = np.append(tmp[columns[i]], arr)

    def mat73(file):
        with h5py.File(file, 'r') as h5:
            data = {k: np.array(v) for k, v in h5.items()}['YearData']
        for i, col in enumerate(data):
            tmp[columns[i]] = np.append(tmp[columns[i]], col)

    print('Loading in Cn2 data')

    columns = ['ts', 'Cn2', 'solar_zenith_angle']
    tmp = {col: np.array([]) for col in columns}
    files = sorted(glob(f'{path}/*.mat'))

    for file in tqdm(files, desc='Files processed'):
        try:
            mat50(file)
        except:
            mat73(file)
    else:
        df = pd.DataFrame(tmp)

    print('Done')
    return df
