"""
Various utility functions
"""
import h5py
import logging
import numpy  as np
import os
import pandas as pd
import sys

from datetime import datetime as dt
from glob     import glob
from mat4py   import loadmat
from tqdm     import tqdm


logging.basicConfig(
    level   = logging.DEBUG,
    format  = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt = '%m-%d %H:%M',
    stream  = sys.stdout
)
Logger = logging.getLogger(os.path.basename(__file__))

def timeit(func):
    """
    Utility decorator to track the processing time of a function

    Parameters
    ----------
    func : function
        The function to track

    Returns
    -------
    any
        Returns the return of the tracked function
    """
    def _wrap(*args, **kwargs):
        start = dt.now()
        ret   = func(*args, **kwargs)
        Logger.debug(f'Finished function {func.__name__} in {(dt.now() - start).total_seconds()} seconds')
        return ret
    # Need to pass the docs on for sphinx to generate properly
    _wrap.__doc__ = func.__doc__
    return _wrap

@timeit
def load_weather(path):
    f"""
    Loads in weather data from .txt files

    Parameters
    ----------
    path : str
        Path to the directory containing weather .txt files found in
        subdirectories per the mapping: {mappings}

    Returns
    -------
    pandas.DataFrame
        A DataFrame of the weather data neatly resampled to one minute intervals
    """
    mappings = {
        'H_PAAVG1M': 'pressure',
        'H_RHAVG1M': 'relative_humidity',
        'H_TAAVG1M': 'temperature',
        'H_WDAVG2M': 'wind_direction',
        'H_WSAVG2M': 'wind_speed',
    }

    Logger.info('Loading in weather data')
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

    return df

@timeit
def load_cn2(path, datenum=False, round=False):
    """
    Loads in Cn2 .mat files, support for mat5.0 and mat7.3 files only

    Parameters
    ----------
    path : str
        Path to the directory containing BLS_[year].mat files
    datenum : bool
        Retains the Matlab datenum column in the return DataFrame, defaults to
        `False` which will drop the column
    round : bool
        Rounds the datetimes that were converted from Matlab datenum to the
        nearest second

    Returns
    -------
    pandas.DataFrame
        A DataFrame of the Cn2 data including solar zenith angle and datetime
        as the index
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

    Logger.info('Loading in Cn2 data')

    columns = ['datenum', 'Cn2', 'solar_zenith_angle']
    tmp = {col: np.array([]) for col in columns}
    files = sorted(glob(f'{path}/BLS_*.mat'))

    for file in tqdm(files, desc='Files processed'):
        try:
            mat50(file)
        except:
            mat73(file)

    # Convert to a DataFrame
    df = pd.DataFrame(tmp)

    # Remove duplicates
    df = df[df.duplicated(keep='first')]

    # Convert Matlab datenum to Python datetime
    df['datetime'] = pd.to_datetime(df.datenum-719529, unit='D')
    df = df.set_index('datetime').sort_index()

    # Round to the nearest second -- cleans up the nanoseconds
    if round:
        df.index = df.index.round('1s')

    # Drop the datenum column
    if not datenum:
        df.drop(columns=['datenum'], inplace=True)

    return df
