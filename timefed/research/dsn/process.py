"""
"""
import argparse
import h5py
import logging
import multiprocessing as mp
import numpy  as np
import pandas as pd
import tsfresh
import warnings

from datetime  import datetime as dtt
from functools import partial
from tables    import NaturalNameWarning
from tqdm      import tqdm

from timefed        import utils
from timefed.config import Config

# Disable h5py warning about setting an integer as a key name
warnings.filterwarnings('ignore', category=NaturalNameWarning)
pd.options.mode.chained_assignment = None

Logger = logging.getLogger('timefed/research/dsn/process.py')

def roll(df, window, step, observations):
    """
    Creates a generator for rolling over a pandas DataFrame with a given window
    size.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame to roll over
    window: str
        The window size to extract
    step: str
        Step size to take when rolling over the DataFrame
    observations: int
        Minimum number of observations required to be a valid window

    Yields
    ------
    pandas.DataFrame
    """
    index   = df.index
    delta   = pd.Timedelta(window)
    zero    = pd.Timedelta(0)
    step    = pd.Timedelta(step)
    size    = df.shape[0] - 1
    windows = []

    # Setup the progress bar
    perc = np.linspace(1, size - observations - 1, 100)
    # bar  = tqdm(total=100, desc='Percent Rolled')
    prog = 0

    i = -1
    while i < size - observations:
        i += 1
        j  = i + observations
        diff = index[j] - index[i]

        # Window too large
        if diff > delta:
            continue
        # Timestamps are not in order causing a negative value
        elif diff <= zero:
            continue

        windows.append(df.iloc[i:j])

        # Incrementally step the progress bar
        # if (i >= perc).sum() > prog:
        #     prog += 1
        #     bar.update()

    return windows

def extract(df, drop, config, features=None):
    """
    """
    df['_ID']   = np.full(len(df), 0)
    df['_TIME'] = df.index
    extracted   = tsfresh.extract_features(
        df.drop(columns=drop),
        column_id    = '_ID',
        column_sort  = '_TIME',
        column_kind  = None,
        column_value = None,
        default_fc_parameters = features,
        disable_progressbar   = True,
        n_jobs = 1
    )

    # Imitate the original index
    extracted.index = [df.index[-1]]

    # Add the dropped columns back in
    extracted[drop] = df[drop].iloc[-1]

    # If this is a classification problem, modify the label
    if config.kind == 'classification':
        if df[config.label].any():
            extracted[config.label] = 1

    return extracted

def add_features(df):
    """
    Adds additional features to a track's dataframe per the config.

    Parameters
    ----------
    df: pandas.DataFrame
        Track DataFrame to add a columns to

    Returns
    -------
    df: pandas.DataFrame
        Modified track DataFrame
    """
    config = Config()

    for feature in config.features.diff:
        df[f'diff_{feature}'] = df[feature].diff()

    return df

def add_label(df, drs):
    """
    Adds the `Label` column to the input DataFrame and marks timestamps as:
         0: Negative class (no DR)
         1: Positive class (had DR)
        -1:  Invalid class (Bad frames)

    Parameters
    ----------
    df: pandas.DataFrame
        Track DataFrame to add a Label column to
    drs: pandas.DataFrame
        The DataFrame containing DR information

    Returns
    -------
    df: pandas.DataFrame
        Modified track DataFrame
    """
    # Create label column with -1 as "bad" rows
    df['Label'] = -1

    # Set rows between B/EoT as 0
    df.Label.loc[df.query('BEGINNING_OF_TRACK_TIME_DT <= RECEIVED_AT_TS <= END_OF_TRACK_TIME_DT').index] = 0

    # Exclude bad frames from the negative class
    df.Label.loc[df.query('TLM_BAD_FRAME_COUNT > 0').index] = -1

    # Verify the bad frames were removed correctly
    assert df.query('Label == 0 and TLM_BAD_FRAME_COUNT > 0').empty, 'Failed to remove bad frames from negative class'

    # Lookup if this track had a DR, if so change those timestamps to 1
    lookup = drs.query(f'SCHEDULE_ITEM_ID == {df.SCHEDULE_ITEM_ID.iloc[0]}')
    if not lookup.empty:
        incident = *timestamp_to_datetime(lookup.INCIDENT_START_TIME_DT), *timestamp_to_datetime(lookup.INCIDENT_END_TIME_DT)

        # Set the incident as positive
        df.Label.loc[df.query('@incident[0] <= RECEIVED_AT_TS <= @incident[1]').index] = 1

    return df

def timestamp_to_datetime(timestamps):
    """
    Converts an integer or floating point timestamp to a Python datetime object.

    Parameters
    ----------
    timestamps: single or iterable of int or float
        A singular or a list of timestamps to convert

    Returns
    -------
    list of or single datetime
    """
    if isinstance(timestamps, (int, float)):
        return dtt.fromtimestamp(timestamps)
    else:
        return [dtt.fromtimestamp(ts) for ts in timestamps]

def decode_strings(df):
    """
    Attempts to apply string.decode() to any column with a dtype of object.

    Parameters
    ----------
    df: pandas.DataFrame
        The DataFrame object to iterate over searching for object dtype columns
        to apply decoding to

    Returns
    -------
    df: pandas.DataFrame
        Same DataFrame object as input but with decoded columns
    """
    for name, column in df.items():
        if column.dtype == 'object':
            try:
                df[name] = column.apply(lambda string: string.decode())
                # Logger.debug(f'Decoded column {name}')
            except:
                Logger.exception(f'Failed to decode column {name}')

    return df

def process(key, config):
    """
    """
    # Read in the DRs
    drs = pd.read_hdf(config.input.drs, key.split('/')[0])
    drs = decode_strings(drs)

    # Read in the track frame
    df = pd.read_hdf(config.input.tracks, key)

    # Decode the strings to cleanup column values (removes the b'')
    df = decode_strings(df)

    # Next attempt to convert DT and TS columns to python DT objects
    for name, column in df.items():
        if 'DT' in name or 'TS' in name:
            df[name] = timestamp_to_datetime(column)

    # Create the label column
    df = add_label(df, drs)

    # Compute additional features
    df = add_features(df)

    # Set the index before saving
    df = df.set_index('RECEIVED_AT_TS')
    df.index.name = 'datetime'

    # Remove rows with invalid data
    df = df.query('Label != -1')      # Bad label
    df = df.dropna(how='any', axis=0) # Has a NaN

    if df.empty:
        return -1

    # Determine which columns to use for processing
    drop = []
    if config.tsfresh:
        drop = [config.label] + list(set(df.columns) - set(config.tsfresh))

    # The label column is always excluded from feature extraction
    if not drop:
        drop = [config.label]

    # Remove columns that are not int or float dtypes
    for col in df:
        dtype = df[col].dtype
        if not np.issubdtype(dtype, int) and not np.issubdtype(dtype, float):
            drop.append(col)

    # Unique the drops list
    drop = list(set(drop))
    # Logger.debug(f'tsfresh will extract only on the following columns: {set(df.columns) - set(drop)}')

    # Create windows for this track
    windows = roll(df,
        window       = config.window,
        step         = config.step,
        observations = config.observations
    )
    extracted = []
    for window in windows:
        window = extract(window, drop, config)
        # Drop columns that have inf values
        window = window.replace([np.inf, -np.inf], np.nan)
        window = window.drop(columns=window.isna().any().index)

        if not window.empty:
            extracted.append(window)

    if extracted:
        # Combine the windows for this track together to save out
        nf = pd.concat(extracted)

        # Save
        df.to_hdf(config.output.tracks, key)
        nf.to_hdf(config.output.windows, key)

        return 0
    return -2

def get_keys(config):
    """
    """
    keys = []
    Logger.debug(f'Reading file: {config.input.tracks}')
    with h5py.File(config.input.tracks, 'r') as h5:
        for sc in h5.keys():
            if config.only.spacecrafts and sc not in config.only.spacecrafts:
                continue

            for ant in h5[sc].keys():
                for track in h5[sc][ant].keys():
                    if track in ['-1.0']:
                        continue

                    for dcc in h5[sc][ant][track].keys():
                        keys.append(f'{sc}/{ant}/{track}/{dcc}')

    yield from tqdm(keys, desc='Keys Processed')

def main():
    """
    """
    config = Config()

    keys = get_keys(config)
    func = partial(process, config=config)
    results = {0: 0, -1: 0, -2: 0}
    with mp.Pool() as pool:
        for result in pool.imap_unordered(func, keys):
            results[result] += 1

    Logger.info(f'{sum(results.values())} tracks were processed')
    Logger.info(f'- Accepted: {results[0]}')
    Logger.info(f'- Rejected:')
    Logger.info(f'  - Empty prior to extraction: {results[-1]}')
    Logger.info(f'  - Empty post extraction    : {results[-2]}')

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/config.yaml',
                                            help     = 'Path to a config.yaml file'
    )
    parser.add_argument('-s', '--section',  type     = str,
                                            default  = 'preprocess',
                                            metavar  = '[section]',
                                            help     = 'Section of the config to use'
    )

    args  = parser.parse_args()
    state = False

    # Initialize the loggers
    utils.init(args)

    # Process
    try:
        state = main()
    except Exception:
        Logger.exception('Caught an exception during runtime')
    finally:
        if state is not None:
            Logger.info('Finished successfully')
        else:
            Logger.info('Failed to complete')
