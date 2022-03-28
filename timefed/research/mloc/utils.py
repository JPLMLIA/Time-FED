"""
Various utility functions
"""
import datetime as dt
import h5py
import logging
import multiprocessing as mp
import multiprocessing.pool as mp_pool
import numpy  as np
import os
import pandas as pd
import seaborn as sns
import sys
import yaml

from datetime import datetime as dtt
from glob     import glob
from mat4py   import loadmat
from tqdm     import tqdm

# Set context of seaborn
sns.set_context('poster', rc={'axes.titlesize': 35, 'axes.labelsize': 30})

Logger = logging.getLogger('timefed/research/mloc/utils.py')

def subselect(args, df):
    """
    Subselects from a dataframe between dates

    Parameters
    ----------
    args : utils.Config
        Config object defining arguments for subselecting
    df : pandas.DataFrame
        The dataframe to subselect from

    Returns
    -------
    sub : pandas.DataFrame
        The subselected dataframe
    """
    # Take a view of the dataframe
    sub = df

    if 'lt' in args:
        Logger.debug(f'\t< {args.lt}')
        sub = sub[sub.index < args.lt]

    if 'gt' in args:
        Logger.debug(f'\t> {args.gt}')
        sub = sub[sub.index > args.gt]

    if 'lte' in args:
        Logger.debug(f'\t<= {args.lte}')
        sub = sub[sub.index <= args.lte]

    if 'gte' in args:
        Logger.debug(f'\t>= {args.gte}')
        sub = sub[sub.index >= args.gte]

    return sub

def cadence(df, limit='30 min', dropna=True):
    """
    Utility statistics function to provide the time deltas that are above and
    below the given limit.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to analyze.
    limit : str
        A delta string to be used in pandas.Timedelta(string).
    dropna : bool
        Drops rows that are NaN for all columns before analyzing. This will
        ensure only cadences between values are provided.
    """
    if dropna:
        df = df.dropna(how='all')

    # Get the cadence of the DataFrame's index
    cadence = pd.Series(df.index).diff().shift(-1)

    # Retrieve the subset that is between 1 minute and the above delta
    between = cadence[
        (pd.Timedelta('1 min') < cadence)
      & (cadence < pd.Timedelta(limit))
     ].value_counts().sort_index()

    # Retrieve the subset that is above the provided delta
    above = cadence[cadence > pd.Timedelta(limit)].sort_values()

    return cadence, between, above

def dense_regions(df, min_size=500):
    """
    Yields subframes of the given dataframe in which all columns are fully
    dense. Yields regions from largest to smallest.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to subset
    min_size : int
        The minimum size of the subframe

    Notes
    -----
    `bounds` is a list of tuples (lower bound, upper bound, size)
    """
    valid = df.notna().all(axis=1)
    bounds = []

    i = j = 0
    while j < valid.size:
        # This row is valid (no nans), increment j
        if valid[j]:
            j += 1
        # Otherwise this row has a nan on it
        else:
            # A window was captured
            if j != i:
                # If the window size is larger than the min_size, save it
                if j-i >= min_size:
                    bounds.append([i, j, j-i])
                # Shift i to j's position
                i = j
            # Otherwise step i, j
            else:
                i += 1
                j += 1
    else:
        # End case, if there was a window at exit check if to save it
        if j != i:
            if j-i >= min_size:
                bounds.append([i, j, j-i])

    # Sort the boundaries by their size
    bounds.sort(reverse=True, key=lambda pair: pair[2])
    for lower, upper, size in bounds:
        try:
            # Check if the subframe is fully dense then yield it
            assert df.iloc[lower:upper].notna().all().all()
            yield df.iloc[lower:upper]
        except:
            print(f'Error: Bounds ({lower}, {upper}, {size}) was not fully dense in the given dataframe. There is a bug in this function, please open a ticket.')

def interpolate(df, limit=30, method='linear', dropna=True, override=False, **kwargs):
    """
    Applies interpolation to a DataFrame up to a given limit. Assumes the
    resolution of the DataFrame is 1 minute and will resample it to insert
    missing timestamps.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to interpolate. Assumes the resolution is 1 minute.
    limit : int
        The limit (in minutes) to interpolate over. Gaps larger than this limit
        will only interpolate up to the limit then leave the NaNs.
    method : str or dict
        The interpolation method to apply. If a string, applies the method to
        all columns. If dict, applies methods on a per-column basis in the form
        {column_name: method}.
    dropna : bool
        Drops rows that are NaN for all columns.
    override : bool
        Overrides safety checks and attempts interpolation anyways.
    **kwargs
        Additional keyword arguments to pass on to the interpolating function.
    """
    if not override:
        # Assert the resolution is 1 minute
        assert df.index.resolution == 'minute'

    # Resample to the minute to insert missing timestamps
    df = df.resample('1 min').mean()

    # Assert the cadence is 1 minute perfect
    cadence = pd.Series(df.index).diff().shift(-1)
    assert cadence.mean() == pd.Timedelta('1 min')

    # Apply interpolation
    if isinstance(method, str):
        df = df.interpolate(method=method, limit=limit, **kwargs)
    else:
        for column, method in method.items():
            df[column] = df[column].interpolate(method=method, limit=limit, **kwargs)

    if dropna:
        df = df.dropna(how='all')

    return df

@timeit
def load_weather(path, interp=False, **interp_args):
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
        # for file in tqdm(files, desc=f'Compiling {column}', position=1):
        for file in files:
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

    if interp:
        df = interpolate(df, **interp_args)

    return df

def datenum2datetime(datenum):
    return dtt.fromordinal(int(datenum)) + dt.timedelta(days=datenum%1) - dt.timedelta(days=366)

@timeit
def load_bls(path, datenum=False, round=False, drop_dups=True, resample=False, resolution='1 min', interp=False, **interp_args):
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
    drop_dups : bool
        Drops rows that are complete duplicates, ie. all columns are the same
        values as another row
    resample : bool
        Resamples the data to the provided resolution using the mean
    resolution : str
        The resample resolution; uses mean() to downsample
    interp : bool
        Enables automatic interpolation of the dataset. Requires resampling the
        data to 1 minute resolution
    **interp_args
        Additionally arguments to pass the the interpolation() function

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

    # Convert Matlab datenum to Python datetime
    df['datetime'] = df.datenum.apply(datenum2datetime)
    df = df.set_index('datetime').sort_index()

    # Round to the nearest second -- cleans up the nanoseconds
    if round:
        df.index = df.index.round('1s')

    # Drop the datenum column
    if not datenum:
        df.drop(columns=['datenum'], inplace=True)

    # Must be resampled if interp is set
    if resample or interp:
        df = df.resample(resolution).mean()

    if interp:
        df = interpolate(df, **interp_args)

    return df

@timeit
def load_r0(path, kind, datenum=False, round=True, drop_dups=True, resample=False, resolution='1 min', interp=False, **interp_args):
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
    drop_dups : bool
        Drops rows that are complete duplicates, ie. all columns are the same
        values as another row
    resample : bool
        Resamples the data to the provided resolution using the mean
    resolution : str
        The resample resolution; uses mean() to downsample
    interp : bool
        Enables automatic interpolation of the dataset. Requires resampling the
        data to 1 minute resolution
    **interp_args
        Additionally arguments to pass the the interpolation() function

    Returns
    -------
    pandas.DataFrame
        A DataFrame of the r0 data
    """
    if kind == 'day':
        cols = ['datenum', 'o(I)_I', 'r0', 'solar_zenith_angle']
    elif kind == 'night':
        cols = ['datenum', 'r0', 'solar_zenith_angle', 'polaris_count']
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
    if drop_dups:
        df = df.drop_duplicates()

    # Drop the datenum column
    if not datenum:
        df.drop(columns=['datenum'], inplace=True)

    # Must be resampled if interp is set
    if resample or interp:
        df = df.resample(resolution).mean()

    if interp:
        df = interpolate(df, **interp_args)

    return df

@timeit
def load_pwv(path, datenum=False, round=True, drop_dups=True, resample=False, resolution='30 min', interp=False, **interp_args):
    """
    Loads the an pwv .txt file

    Parameters
    ----------
    path : str
        Location of PVWwithWeather.txt
    datenum : bool
        Retains the datenum column in the return DataFrame, defaults to
        `False` which will drop the column
    round : bool
        Rounds the datetimes that were converted from Matlab datenum to the
        nearest second
    drop_dups : bool
        Drops rows that are complete duplicates, ie. all columns are the same
        values as another row
    resample : bool
        Resamples the data to the provided resolution using the mean
    resolution : str
        The resample resolution; uses mean() to downsample
    interp : bool
        Enables automatic interpolation of the dataset. Requires resampling the
        data to 1 minute resolution
    **interp_args
        Additionally arguments to pass the the interpolation() function

    Returns
    -------
    pandas.DataFrame
        A DataFrame of the r0 data
    """
    cols = ['datenum', 'water_vapor', 'wind_direction', 'wind_speed', 'temperature', 'humidity', 'wind_gust', 'pressure', 'dewpoint']

    df = pd.read_csv(path, names=cols)

    # Convert Matlab datenum to Python datetime
    df['datetime'] = df.datenum.apply(datenum2datetime)
    df = df.set_index('datetime').sort_index()

    # Round to the nearest second -- cleans up the nanoseconds
    if round:
        df.index = df.index.round('1s')

    # Drop duplicates
    if drop_dups:
        df = df.drop_duplicates()

    # Drop the datenum column
    if not datenum:
        df.drop(columns=['datenum'], inplace=True)

    # Must be resampled if interp is set
    if resample or interp:
        df = df.resample(resolution).mean()

    if interp:
        df = interpolate(df, **interp_args)

    return df

def compile_datasets(weather=None, bls=None, r0_day=None, r0_night=None, h5='', resample='5 min', smooth=['r0', 'r0_day', 'r0_night']):
    """
    Ingests all of the raw data into dataframes then compiles them into a merged
    dataframe.

    Parameters
    ----------
    weather : str
        Equivalent to the `path` argument of utils.load_weather()
    bls : str
        Equivalent to the `path` argument of utils.load_bls()
    r0_day : str
        Equivalent to the `path` argument of utils.load_r0()
    r0_night : str
        Equivalent to the `path` argument of utils.load_r0()
    h5 : str
        Optional path to an h5 to write to
    resample : str
        The rate to resample the merged dataframe
    smooth : list of str
        List of columns to apply smoothing on
    """
    # Check if the incoming h5 has the data already, skip loading from raw
    has_keys = False
    if os.path.exists(h5):
        with h5py.File(h5, 'r') as f:
            has_keys = all([
                'r0/day'   in f,
                'r0/night' in f,
                'bls'      in f,
                'weather'  in f,
            ])

    Logger.debug('Loading data')
    # Load in the data
    if has_keys:
        Logger.debug('Loading datasets from h5')
        data = {
            'r0/day'  : pd.read_hdf(h5, 'r0/day'),
            'r0/night': pd.read_hdf(h5, 'r0/night'),
            'bls'     : pd.read_hdf(h5, 'bls'),
            'weather' : pd.read_hdf(h5, 'weather')
        }
    else:
        Logger.debug('Loading datasets from their raw files')
        data = {
            'r0/day'  : load_r0(r0_day,   kind='day',   round=True, resample=False, datenum=False),
            'r0/night': load_r0(r0_night, kind='night', round=True, resample=False, datenum=False),
            'bls'     : load_bls(bls, round=True, resample=False, datenum=False),
            'weather' : load_weather(weather)
        }
        # Postprocess
        data['r0/day']['r0'] *= 100 # Convert to centimeters
        data['r0/night'].drop(columns='polaris_count', inplace=True)

        # Save individual frames
        if h5:
            for key, df in data.items():
                df.to_hdf(h5, key)

    Logger.debug('Merging dataframes together')
    # Merge the frames together
    df = pd.merge(data['r0/day'], data['r0/night'], how='outer', suffixes=['_day', '_night'], on=['datetime', 'solar_zenith_angle'])
    df = pd.merge(df, data['bls'], how='outer', on=['datetime', 'solar_zenith_angle'])
    df = pd.merge(data['weather'], df, how='outer', on='datetime')

    # Sort the datetime index
    df.sort_index(inplace=True)

    # Create the r0 column by merging day and night
    df['r0'] = df.r0_day.combine_first(df.r0_night)

    # Apply smoothing
    for col in smooth:
        if col == 'r0':
            continue # Skip r0 to do separately
        elif col in ['Cn2', 'r0_night']:
            Logger.debug(f'Smoothing {col} with 2 minimum observations')
            df[f'{col}_10T'] = df[col].rolling('10 min', min_periods=2).median()
        else:
            # hardcoded 10 minute smoothing with a minimum of 20% observations (assuming seconds) 10*60*20%=120
            Logger.debug(f'Smoothing {col} with 120 minimum observations')
            df[f'{col}_10T'] = df[col].rolling('10 min', min_periods=120).median()

    # Smoothed r0 needs to be the merge of the smoothed day and night rather than a smooth on r0 merged
    #  due to observation requirements causing night to become fully NaN
    if 'r0' in smooth:
        Logger.debug('Creating smoothed r0 merged separately')
        # If already smoothed, use columns otherwise do smoothing
        if 'r0_day_10T' in df and 'r0_night_10T' in df:
            df['r0_10T'] = df.r0_day_10T.combine_first(df.r0_night_10T)
        else:
            df['r0_10T'] = df['r0_day'].rolling('10 min', min_periods=120).median().combine_first(
                df['r0_night'].rolling('10 min', min_periods=2).median()
            )

    Logger.debug(f'Resampling to {resample}')
    # Resample with at least 2 observations
    minobs = lambda s: np.nan if len(s) < 2 else s.median()
    df = df.resample(resample).median()#.apply(minobs)

    # Save merged
    if h5:
        df.to_hdf(h5, 'merged')

    return df

def compile_pwv(path, h5=None, resample='30 min'):
    """
    Ingests all of the raw data into dataframes then compiles them into a merged
    dataframe.

    Parameters
    ----------
    h5 : str
        Optional path to an h5 to write to
    resample : str
        The rate to resample the merged dataframe
    smooth : list of str
        List of columns to apply smoothing on
    """

    Logger.debug('Loading datasets from their raw files')
    # Load in the data
    df = load_pwv(path)

    # Save individual frames
    if h5 is not None:
        df.to_hdf(h5, 'pwv')

    # Sort the datetime index
    df.sort_index(inplace=True)

    Logger.debug(f'Resampling to {resample}')
    df = df.resample(resample).median()#.apply(minobs)

    # Save merged
    df.to_hdf(h5, 'pwv')

    return df

class _Helper:
    """
    Helper object for Config to allow for nested dot notation accessing config
    args
    """
    def __init__(self, data):
        self.__dict__ = data
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, _Helper(value))
            else:
                setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def __iter__(self):
        return iter(self.__dict__.items())

    def __getitem__(self, key):
        return getattr(self, key)

class Config:
    def __init__(self, file, section):
        """
        Loads a config.yaml file

        Parameters
        ----------
        path : str
            Path to the config.yaml file to read
        section : str
            Selects which section of the config to return
        """
        with open(file, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        self.config = _Helper(data)
        self.active = getattr(self.config, section)

    def __getattr__(self, key):
        try:
            return getattr(self.active, key)
        except:
            return None

    def __contains__(self, key):
        return hasattr(self, key)

    def __getitem__(self, key):
        return self.__getattr__(key)

class NoDaemonProcess(mp.Process):
    '''
    Credit goes to Massimiliano
    https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
    '''
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonContext(type(mp.get_context())):
    '''
    Credit goes to Massimiliano
    https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
    '''
    Process = NoDaemonProcess

class Pool(mp_pool.Pool):
    '''
    Credit goes to Massimiliano
    https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
    '''
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super().__init__(*args, **kwargs)
