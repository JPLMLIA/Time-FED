"""
Various utility functions
"""
import datetime as dt
import h5py
import logging
import numpy  as np
import os
import pandas as pd
import sys

from datetime import datetime as dtt
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
        start = dtt.now()
        ret   = func(*args, **kwargs)
        Logger.debug(f'Finished function {func.__name__} in {(dtt.now() - start).total_seconds()} seconds')
        return ret
    # Need to pass the docs on for sphinx to generate properly
    _wrap.__doc__ = func.__doc__
    return _wrap

@timeit
def load_weather(path, interpolate=True, **interp_args):
    """
    Loads in weather data from .txt files

    Parameters
    ----------
    path : str
        Path to the directory containing weather .txt files found in
        subdirectories per the mappings

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
                pd.read_csv(file, sep='\t', header=None, names=['datetime', column], index_col='datetime', parse_dates=True, dtype={column: float}, na_values='///')
            ])
        else:
            _df = _df.resample('1T').mean()
            if df is not None:
                df = df.join(_df, sort=True)
            else:
                df = _df

    if interpolate:
        df.interpolate(inplace=True, **interp_args)

    return df

def datenum2datetime(datenum):
    return dtt.fromordinal(int(datenum)) + dt.timedelta(days=datenum%1) - dt.timedelta(days=366)

@timeit
def load_bls(path, datenum=False, round=False, drop_dups=True):
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
        data = list(loadmat(file).values())[0]
        return pd.DataFrame(data, columns=cols)

    def mat73(file):
        with h5py.File(file, 'r') as h5:
            data = h5[list(h5.keys())[0]]
            ret = pd.DataFrame(data).T
            return ret.rename(columns={i: cols[i] for i in range(len(cols))})

    Logger.info('Loading in Cn2 data')

    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = sorted(glob(f'{path}/BLS_*.mat'))
    else:
        Logger.error(f'The provided path is neither a directory nor a file: {path}')
        return

    cols = ['datenum', 'Cn2', 'solar_zenith_angle']
    df   = pd.DataFrame(columns=cols)
    for file in tqdm(files, desc='Files processed'):
        try:
            df = df.append(mat50(file))
        except:
            df = df.append(mat73(file))

    # Remove duplicates
    if drop_dups:
        df = df.drop_duplicates()

    df['datetime'] = df.datenum.apply(datenum2datetime)

    # Convert Matlab datenum to Python datetime
    #df['datetime'] = pd.to_datetime(df.datenum-719529, unit='D')

    df = df.set_index('datetime').sort_index()

    # Round to the nearest second -- cleans up the nanoseconds
    if round:
        df.index = df.index.round('1s')

    # Drop the datenum column
    if not datenum:
        df.drop(columns=['datenum'], inplace=True)

    return df

def load_r0(path, kind, datenum=False, round=True, drop=True):
    """
    Loads the an r0 .mat file, specifically:

    r0 daytime   = fl4.mat
    r0 nighttime = Cyclops1820.mat

    Parameters
    ----------
    path : str
        Location of fl4.mat
    kind : str
        Selects which kind of dataset the provided file is, either "day" or
        "night"
    datenum : bool
        Retains the Matlab datenum column in the return DataFrame, defaults to
        `False` which will drop the column
    round : bool
        Rounds the datetimes that were converted from Matlab datenum to the
        nearest second
    drop : bool
        Drops rows that are complete duplicates, ie. all columns are the same
        values as another row

    Returns
    -------
    pandas.DataFrame
        A DataFrame of the r0 data
    """
    if kind == 'day':
        cols = ['datenum', 'o(I)_I', 'r0', 'sun_zenith_angle']
    elif kind == 'night':
        cols = ['datenum', 'r0', 'sun_zenith_angle']
    else:
        Logger.error('load_r0() requires the `kind` parameter to be either "day" or "night"')
        return

    data = list(loadmat(path).values())[0]
    df = pd.DataFrame(data, columns=cols)

    # Convert Matlab datenum to Python datetime
    df['datetime'] = df.datenum.apply(datenum2datetime)
    df = df.set_index('datetime').sort_index()

    # Round to the nearest second -- cleans up the nanoseconds
    if round:
        df.index = df.index.round('1s')

    # Drop duplicates
    if drop:
        df = df.drop_duplicates()

    # Drop the datenum column
    if not datenum:
        df.drop(columns=['datenum'], inplace=True)

    return df
