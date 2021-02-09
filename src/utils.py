"""
Various utility functions
"""
import datetime as dt
import h5py
import logging
import numpy  as np
import os
import pandas as pd
import seaborn as sns
import sys

from datetime import datetime as dtt
from glob     import glob
from mat4py   import loadmat
from tqdm     import tqdm

# Set context of seaborn
sns.set_context('talk')

logging.basicConfig(
    level   = logging.DEBUG,
    format  = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt = '%m-%d %H:%M',
    stream  = sys.stdout
)
Logger = logging.getLogger(os.path.basename(__file__))

# Increase matplotlib's logger to warning to disable the debug spam it makes
logging.getLogger('matplotlib').setLevel(logging.WARNING)

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

def compile_datasets(weather=None, bls=None, r0_day=None, r0_night=None, h5=None, resample='median'):
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
        If set, resamples the dataframes to 1 minute using the method given
    """
    # Import here to prevent being a package dependency
    import xarray as xr

    data = {}

    if r0_day:
        data['r0/day']   = load_r0(r0_day,   kind='day',   round=True, resample=False, datenum=False)
        data['r0/day']['r0'] *= 100
    if r0_night:
        data['r0/night'] = load_r0(r0_night, kind='night', round=True, resample=False, datenum=False)
        data['r0/night'].drop(columns='polaris_count', inplace=True)
    if weather:
        data['weather'] = load_weather(weather)
    if bls:
        data['bls'] = load_bls(bls, round=True, resample=False, datenum=False)

    # Resample to a minute
    if resample:
        for key, df in data.items():
            if resample == 'median':
                data[key] = df.resample('1 min').median()
            elif resample == 'mean':
                data[key] = df.resample('1 min').mean()

    # Save dataframes
    if h5:
        for key, df in data.items():
            df.to_hdf(h5, key)

    # Convert to xarray
    for key, df in data.items():
        data[key] = df.to_xarray()

    # Combine r0 day and night
    if 'r0/day' in data and 'r0/night' in data:
        data['r0'] = xr.merge([
            data['r0/day'],
            data['r0/night']
        ])
        data['r0/day']   = data['r0/day'].rename({'r0': 'r0_day'})
        data['r0/night'] = data['r0/night'].rename({'r0': 'r0_night'})

    # Reorganize so r0 comes before bls
    dss = []
    for key, ds in data.items():
        if key in ['r0']:
            dss = [ds] + dss
        else:
            dss.append(ds)

    ds = xr.merge(dss, compat='override')
    df = ds.to_dataframe()

    if h5:
        df.to_hdf(h5, 'merged')

    return data, dss, df
