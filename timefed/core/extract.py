import argparse
import logging
import re

import h5py
import numpy  as np
import pandas as pd
import ray
import tsfresh

from mlky import (
    Config,
    Sect
)
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from tsfresh.feature_extraction import ComprehensiveFCParameters

from timefed.utils import utils


Logger = logging.getLogger('timefed/core/extract.py')


def report(stats):
    """
    Reports the statistics gathered by the roll function

    Parameters
    ----------
    stats: mlky.Sect
        Sect object produced by roll()
    """
    Logger.info('Roll stats:')
    Logger.info(f'- Frequency of the data is: {stats.frequency}')
    Logger.info(f'- The data ranges over {stats.range}')
    Logger.info(f'- Using a window size of {stats.window} and a step of {stats.step}, the size of each window is {stats.size} samples')
    Logger.info(f'- Windows produced:')
    Logger.info(f'-- Total possible : {stats.possible}')

    if stats.possible > 0:
        Logger.info(f'-- Number accepted: {stats.valid} ({stats.valid/stats.possible:.2%}%)')

        if stats.optional:
            Logger.info(f'-- Number of windows containing each optional variable:')
            utils.align_print(stats.optional, print=Logger.info, prepend='--- ')

        if stats.possible != stats.valid:
            Logger.info(f'-- Number rejected: {stats.possible-stats.valid} ({(stats.possible-stats.valid)/stats.possible:.2%}%)')
            Logger.info(f'-- Reasons for rejection:')
            utils.align_print(stats.reasons, print=Logger.info, prepend='--- ')


def roll(df, window, frequency, step=1, required=None, optional=[], as_frames=False):
    """
    Creates a generator for rolling over a pandas DataFrame with a given window
    size.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame to roll over
    window: str
        The window size to extract
    frequency: str
        The frequency of the input data
    step: str or int, default=1
        Step size to take when rolling over the DataFrame
        If int, steps that many indices
        If str, uses time to determine next index
    required: list, default=None
        Columns to require
    optional: list, default=[]
        Optional columns
    as_frames: bool, default=True
        Returns the windows as pandas DataFrames

    Returns
    -------
    windows: list
        List of pairs (i, j) for the start and end indices
        for a valid window

    Notes
    -----
    The index must be a pandas.TimeIndex. Does not support pandas.PeriodIndex.
    """
    stats = Sect({
        'possible': 0,
        'valid'   : 0,
        'optional': {},
        'reasons' : {
            'wrong_size'   : 0,
            'required_nans': 0,
            'had_gap'      : 0,
            'not_ordered'  : 0
        }
    })

    zero = pd.Timedelta(0)
    freq = (df.index[1:] - df.index[:-1]).value_counts().sort_values(ascending=False)
    if zero in freq:
        Logger.warning('Duplicate timestamps were detected, windowing may return unexpected results')

    if frequency is None:
        frequency = freq.index[0]
        Logger.info(f'Frequency not provided, selecting the most common frequency difference: {frequency}')

    frequency = stats.frequency = pd.Timedelta(frequency)

    def _step_by_time(i):
        k = df.index[i] + offset
        while df.index[i] < k:
            i += 1
        return i

    def _step_by_index(i):
        return i + offset

    if isinstance(step, str):
        offset = pd.Timedelta(step)
        step   = _step_by_time

    elif isinstance(step, int):
        offset = step
        step   = _step_by_index

    delta = pd.Timedelta(window)
    size  = stats.size = int(delta / frequency)
    if size < 1:
        Logger.error(f'The window size is too short for the cadence of the data: size = int(delta / frequency) = int({delta} / {frequency}) = {size}')
        return [], stats

    if not required:
        required = list(df.columns)

    if not optional:
        optional = set(df.columns) - set(required)

    for column in optional:
        stats.optional[column] = 0

    stats.window = delta
    stats.step   = offset
    stats.range  = df.index[-1] - df.index[0]

    samples = df.shape[0]
    windows = []

    i = 0
    while i <= samples - size:
        stats.possible += 1
        j = i
        k = j + size
        i = step(i)

        window = df.iloc[j:k]

        # This window was the wrong size (rare edge case)
        if window.shape[0] != size:
            stats.reasons.wrong_size += 1
            continue

        # This window had NaNs in a required column
        if window[required].isna().any(axis=None):
            stats.reasons.required_nans += 1
            continue

        diff = window.index[-1] - window.index[0]

        # Window too large
        if diff > delta:
            stats.reasons.had_gap += 1
            continue
        # Timestamps are not in order causing a negative value or there's duplicates
        elif diff <= zero:
            stats.reasons.not_ordered += 1
            continue

        stats.valid += 1
        windows.append((j, k))

        for column in optional:
            if not window[column].isna().any():
                stats.optional[column] += 1

    if as_frames:
        windows = [df.iloc[i:j] for i, j in windows]

    return windows, stats


def get_features(whitelist=None, blacklist=None, interactive=False):
    """
    Retrieves the dictionary of tsfresh features to use and their arguments. Can be
    used in an interactive mode or configured via the configuration file.

    Parameters
    ----------
    whitelist : list, default=None
        List of feature names to only include for calculations
    blacklist : list, default=None
        List of feature names to exclude from calculations
    interactive : bool, default=False
        Enables interactive mode for this function, printing the features list to
        screen and prompting for a list to use for calculations

    Returns
    -------
    features : dict
        Dictionary of {feature_name: feature_arguments} for tsfresh to use during
        feature extraction
    """
    features = ComprehensiveFCParameters()

    # Apply white/black lists if available
    if whitelist:
        features = {key: value for key, value in features.items() if key in whitelist}
    if blacklist:
        features = {key: value for key, value in features.items() if key not in blacklist}

    # Prompt the user with a list of available features
    if interactive:
        retain = list(features.keys())
        Logger.info('Current list of features to be used in feature extraction:')

        for i, feat in enumerate(retain):
            Logger.info(f'\t{i}\t- {feat}')

        response = input('Please select which features to use in extraction (eg. 0 3 11): ')
        indices  = re.findall(r'(\d+)', response)
        retain   = [retain[int(i)] for i in indices]

        features = {key: value for key, value in features.items() if key in retain}

    return features


def verify(df):
    """
    Verifies an extracted DataFrame is valid for the next steps of TimeFED by checking:
    - If any column has NaNs, remove it
    - If any column has Infs, remove it

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to verify on

    Returns
    -------
    df : pd.DataFrame
        Same input DataFrame with invalid columns removed
    """
    # Drop columns that have any NaNs
    nans = df.isna().sum()
    nans = nans[nans > 0].sort_values()
    if nans.any():
        Logger.info(f'{nans.shape[0]}/{df.shape[1]} ({nans.shape[0]/df.shape[1]*100:.2f}%) columns had NaNs in them and will be dropped, see debug for more information')

        Logger.debug('Columns with NaN values:')
        utils.align_print(nans, prepend='- ', print=Logger.debug)

        df = df.drop(columns=nans.index)

    # Drop columns that have inf values
    df = df.replace([np.inf, -np.inf], np.nan)
    inf_cols = df.columns[df.isna().any()]
    if inf_cols.any():
        Logger.info(f'{len(inf_cols)} columns with infinity values found and will be dropped, see debug for more information')
        Logger.debug('Columns with inf values:')
        for col in inf_cols:
            Logger.debug(f'- {col}')

        df = df.drop(columns=inf_cols)

    return df


@ray.remote
def extract(df, slice=None, columns=[], target=None, features=None, index=-1, classification=False):
    """
    Prepares a window of data and performs tsfresh feature extraction

    Parameters
    ----------
    df: pandas.core.DataFrame
        Either a window of data or the full DataFrame itself to be
        subsetted using a provided slice
    slice: slice, default=None
        If given, uses this slice on df as the window
    columns: list, default=[]
        Columns to process through tsfresh
    target: str, default=None
        This is the target variable
    features: dict, default=None
        Subset of tsfresh.feature_extraction.ComprehensiveFCParameters
        None uses all feature functions with default params
    index: int, default=-1
        Index set the window at. Defaults to -1 which is the last
        index of the window

    Returns
    -------
    extracted: pandas.core.DataFrame

    Notes
    -----
    Requires a minimum of two columns: a target column and a data column. There must
    always be 1 target column, and there can be N>0 columns for data.
    """
    # TODO: https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
    # pd.options.mode.chained_assignment = None

    if slice:
        df = df.iloc[slice, :].copy()

    if not columns:
        columns = list(df.columns)

    columns += ['_ID', '_TIME']

    if target in columns:
        columns.remove(target)

    df['_ID']   = np.full(len(df), 0)
    df['_TIME'] = df.index

    try:
        extracted = tsfresh.extract_features(
            df[columns],
            column_id    = '_ID',
            column_sort  = '_TIME',
            column_kind  = None,
            column_value = None,
            default_fc_parameters = features,
            disable_progressbar   = True,
            n_jobs = 1
        )
    except Exception as e:
        Logger.debug(f'Failed to process a window slice {slice}: {e}')
        return str(e)

    # Imitate the original index
    idx = df.index[index]
    extracted.index = [idx]

    # Add the excluded columns back in
    excluded = list(set(df.columns) - set(columns))
    if excluded:
        extracted.loc[idx, excluded] = df[excluded].iloc[index]

    # If this is a classification problem, modify the target
    if classification and df[target].any():
        extracted[target] = 1

    return extracted


@ray.remote
def rotate(df, slice=None, columns=[], index=-1, **kwargs):
    """
    Rotates a DataFrame by stacking columns and converting to a single row DataFrame.
    Sets the value of the last index as the index of the rotated DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to be rotated.
    slice : slice, default=None
        If given, uses this slice on df as the window.
    columns: list, default=[]
        Columns to rotate, all others will be re-added as the value from the specified index
    index : int, default=-1
        Index set the window at. Defaults to -1 which is the last index of the window.
    **kwargs
        Ignore any additional key-word arguments to enable compatibility with other
        extraction functions that may have alternative parameters

    Returns
    -------
    pandas.DataFrame
        Rotated DataFrame with a single row.

    Example
    -------
    >>> import pandas as pd
    >>> data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
    >>> df = pd.DataFrame(data, index=[-1, -2, -3])
    >>> rotated_df = rotate(df)
    >>> print(rotated_df)
        A_0  A_1  A_2  B_0  B_1  B_2  C_0  C_1  C_2
    -3    1    2    3    4    5    6    7    8    9
    """
    if slice:
        df = df.iloc[slice, :].copy()

    if not columns:
        columns = list(df.columns)

    # Retrieve what will become the single index of this window
    idx = df.index[index]

    # For each column, extract it and rename the index to "[column name]_[index position]"
    hold = []
    for col, data in df[columns].items():
        data.index = [f'{col}_{i}' for i in range(data.size)]
        hold.append(data)

    # Concatenate the columns together and rotate
    rotated = pd.concat(hold).to_frame().T
    rotated.index = [idx]

    # Add the excluded columns back in
    excluded = list(set(df.columns) - set(columns))
    if excluded:
        rotated.loc[idx, excluded] = df[excluded].iloc[index]

    return rotated


class Extract:
    def __init__(self):
        """
        """
        self.C = Config.extract
        self.model_kind   = Config.model.kind
        self.model_target = Config.model.target or None

        if self.C.ray:
            ray.init(**self.C.ray)

        # Determine what keys from the H5 file to load
        match (keys := self.C.multi):
            # Retrieve all subkeys from this group
            case str():
                with h5py.File(Config.extract.file, 'r') as h5:
                    keys = [f'{keys}/{key}' for key in h5[keys].keys()]

                self.multi = True
                self.metadata = {}

            # Process these specific keys
            case list():
                self.multi = True
                self.metadata = {}

            # Single track case
            case _:
                keys = ['preprocess/complete']
                self.multi = False
                self.metadata = None

        # Select a window processing method
        match self.C.method:
            case "tsfresh":
                process = extract
                params  = {
                    'features'      : ray.put(get_features(**self.C.features)),
                    'target'        : ray.put(self.model_target),
                    'columns'       : ray.put(self.C.get('columns', [])),
                    'index'         : ray.put(self.C.get('index'  , -1)),
                    'classification': ray.put(self.model_kind == 'classification')
                }
            case "rotate":
                process = rotate
                params  = {
                    'columns': ray.put(self.C.get('columns', [])),
                    'index'  : ray.put(self.C.get('index'  , -1)),
                }
            case invalid:
                raise AttributeError(f"Invalid method chosen: {invalid}")

        # Perform the processing via Ray
        for key in tqdm(keys, position=1, desc='Processing Frames'):
            df = self.process(key, process, params)

        # Save the metadata, if there was any
        if self.model_kind == 'classification' and self.multi:
            if (file := Config.subselect.metadata):
                if self.metadata:
                    utils.save_pkl(file, self.metadata)
                else:
                    Logger.error('No metadata produced for a multi-track classification case, this may have consequences down the line')
            else:
                Logger.error('Classification runs must define Config.subselect.metadata')


    def process(self, key, func, params):
        """
        """
        df = pd.read_hdf(self.C.file, key)

        windows, stats = roll(df, as_frames=False, **self.C.roll)
        report(stats)

        if stats.valid == 0:
            Logger.error('No windows were accepted for this track of data. Nothing to do, returning nothing')
            return

        if self.C.flush:
            Logger.debug(f'Window flushing enabled, will be written to: {self.C.file}[extract/windows/w[i]]')

        # Place constant params into shared memory
        df_id = ray.put(df)
        jobs  = [
            func.remote(df=df_id, slice=slice(*window), **params)
            for window in windows
        ]

        # Make sure any previous runs' data aren't still lingering
        self.clear_flush()

        errored = Sect(count=0, reasons={})
        returns = []
        for i in tqdm(range(len(windows)), desc='Processing Windows', position=0):
            [done], jobs = ray.wait(jobs, num_returns=1)
            window = ray.get(done)

            # Track reasons for errors
            if isinstance(window, str):
                errored.count += 1

                if window not in errored.reasons:
                    errored.reasons[window] = 0

                errored.reasons[window] += 1
                continue

            # Flush to disk, if set
            if self.C.flush:
                window.to_hdf(self.C.file, key=f'extract/windows/w{i}')
            else:
                returns.append(window)

            del window, done

        # Delete from ray memory
        del df, df_id

        if errored:
            Logger.warning(f'{errored.count} windows failed')

            Logger.debug(f'Reasons:')
            for reason, count in errored.reasons.items():
                Logger.debug(f'- {count} = {reason}')

            if errored.count == len(windows):
                Logger.error('All windows failed, returning nothing')
                return

        if self.C.flush:
            Logger.info('Loading windows into memory')
            with h5py.File(Config.extract.file, 'r') as h5:
                keys = list(h5['extract/windows'])

            # Pull into memory
            returns = [pd.read_hdf(Config.extract.file, f'extract/windows/{key}') for key in keys]

        Logger.info(f'Concatenating {len(returns)} window frames together')
        if returns:
            df = pd.concat(returns).sort_index()
            df = verify(df)
        else:
            Logger.error('No window frames were gathered, returning nothing')
            return

        if self.multi:
            counts = {}
            if self.model_kind == 'classification':
                counts = df[self.model_target].value_counts()

            self.metadata[key] = {
                'start': df.index[0],
                'end'  : df.index[-1],
                'neg'  : counts.get(0, 0),
                'pos'  : counts.get(1, 0)
            }

            df.to_hdf(self.C.file, key=f'extract/tracks/{key}')
        else:
            Logger.info('Saving to key extract/complete')
            df.to_hdf(self.C.file, key=f'extract/complete')


    def clear_flush(self):
        """
        Clears the [extract/windows] key from the H5 file
        """
        with h5py.File(self.C.file, 'a') as h5:
            if 'extract/windows' in h5:
                del h5['extract/windows']


@utils.timeit
def main():
    """
    """
    Extract()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--Config',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/Config.extract.yaml',
                                            help     = 'Path to a Config.extract.yaml file'
    )
    parser.add_argument('-s', '--section',  type     = str,
                                            default  = 'extract',
                                            help     = 'Section of the Config to use'
    )

    utils.init(args)

    try:
        with logging_redirect_tqdm():
            main()

        Logger.info('Finished')
    except Exception as e:
        Logger.exception('Failed to complete due to an exception')
