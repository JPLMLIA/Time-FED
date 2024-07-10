"""
"""
import logging
import os
import sys

from datetime import datetime as dtt

import h5py
import numpy as np
import pandas as pd
import ray
import tsfresh

from mlky import (
    Config,
    Sect
)
from mlky.utils.track import Track
from tqdm import tqdm
from tsfresh.feature_extraction import ComprehensiveFCParameters

from timefed.utils import utils


sys.setrecursionlimit(5_000)
Logger = logging.getLogger('timefed/extract')


class Roll:
    zero = pd.Timedelta(0)

    def __init__(self, df, window, frequency=None, step=1, required=None, optional=[], method='groups'):
        """
        Performs some preprocessing for the roll function.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to operate on. The index must be a datetime.
        window : str
            The window size to extract. This must be a pandas.Timedelta compatible
            string, such as 5m or 2h.
        frequency : str, default=None
            The assumed frequency of the DataFrame. Must be a pandas.Timedelta
            compatible string. If not provided, will assume the most common frequency.
        step : int or str, default=1
            The step size between windows. If int, steps by index position. If str,
            uses pandas.Timedelta to step along the index.
        required : list, default=None
            These columns are required to be fully dense in each window.
        optional : list, default=[]
            These columns are optional. (NYI)
        method : 'groups', 'pandas', None
            How to format the .windows attribute
            None:
                Leave windows in index form tuples [(i, j), ...] such that i is the
                start of the window and j is the end on the index
            pandas:
                Convert windows to individual pandas DataFrames (view, not copy)
            groups:
                Does 'pandas' then copies the frames, adds a windowID column, then
                stacks them into a single DataFrame

        Notes
        -----
        >>> windows = Roll(df, '60s')
        >>> windows.roll()
        20
        >>> windows.windows
        ...
        """
        self.df = df.copy()
        self.windows = []
        self.method  = method

        freqs = (df.index[1:] - df.index[:-1]).value_counts().sort_values(ascending=False)
        if self.zero in freqs:
            Logger.info('Duplicate timestamps were detected, windowing may return unexpected results')

        if frequency is None:
            frequency = freqs.index[0]
            Logger.info(f'Frequency not provided, selecting the most common frequency difference: {frequency}')

        self.freq = pd.Timedelta(frequency)

        if isinstance(step, str):
            self.offset = pd.Timedelta(step)
            self.step   = self.stepByTime

        elif isinstance(step, int):
            self.offset = step
            self.step   = self.stepByIndex

        self.delta = pd.Timedelta(window)
        self.size  = int(self.delta / self.freq)
        if self.size < 1:
            Logger.error(f'The window size is too short for the cadence of the data (min size 1): size = int(delta / frequency) = int({delta} / {frequency}) = {size}')

        if not required:
            required = list(df.columns)
        self.required = required

        if not optional:
            optional = set(df.columns) - set(self.required)
        self.optional = optional

        # Stats
        self.possible = 0
        self.valid    = 0
        self.reasons  = Sect(
            wrong_size  = 0,
            had_nans    = 0,
            had_gap     = 0,
            not_ordered = 0
        )


    def stepByTime(self, i):
        """
        Performs a step in time that respects an imperfectly sampled datetime index

        Parameters
        ----------
        i : int
            Index position to start from

        Returns
        -------
        i : int
            The next index position such that this index is still less than [i] + offset
        """
        k = self.df.index[i] + self.offset
        while self.df.index[i] < k:
            i += 1
        return i


    def stepByIndex(self, i):
        """
        Performs a step by integer index

        Parameters
        ----------
        i : int
            Step from this

        Returns
        -------
        int
            i + step offset
        """
        return i + self.offset


    def roll(self):
        """
        Rolls over a datetime index and calculates the possible valid windows that can
        be extracted. After executing, access the windows from the .windows attribute.

        Returns
        -------
        self.valid : int
            The number of valid windows available
        """
        total = self.df.shape[0] - self.size
        track = Track(total, step=10, print=Logger.info)

        i = 0
        while i <= total:
            self.possible += 1

            j = i
            k = j + self.size
            i = self.step(i)
            track(i)

            window = self.df.iloc[j:k]

            # This window was the wrong size (rare edge case)
            if window.shape[0] != self.size:
                self.reasons.wrong_size += 1
                continue

            # This window had NaNs in a required column
            if window[self.required].isna().any(axis=None):
                self.reasons.had_nans += 1
                continue

            diff = window.index[-1] - window.index[0]

            # Window too large
            if diff > self.delta:
                self.reasons.had_gap += 1
                continue
            # Timestamps are not in order causing a negative value or there's duplicates
            elif diff <= self.zero:
                self.reasons.not_ordered += 1
                continue

            self.valid += 1
            self.windows.append((j, k))

        if self.method in ['pandas', 'groups']:
            self.windows = [self.df.iloc[i:j] for i, j in self.windows]

            if self.method == 'groups':
                Logger.info('Duplicating and stacking windows to create groups')
                for w, df in enumerate(self.windows):
                    self.windows[w] = df.copy()
                    self.windows[w]['windowID'] = w

                self.windows = pd.concat(self.windows)

        return self.valid


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
        Enables interactive mode for this function, Logger.infoing the features list to
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


def extract(window, columns=[], index=-1, features=None, target=None, classification=False):
    """
    Prepares a window of data and performs tsfresh feature extraction

    Parameters
    ----------
    window : dict
        Dict in format {column: np.array}
    columns : list, default=[]
        Columns to process through tsfresh
    index : int, default=-1
        Index set the window at. Defaults to -1 which is the last
        index of the window
    features : dict, default=None
        Subset of tsfresh.feature_extraction.ComprehensiveFCParameters
        None uses all feature functions with default params
    target : str, default=None
        This is the target variable
    classification : bool, default=False
        If true, changes the target to 1 if any in the window

    Returns
    -------
    extracted: pandas.core.DataFrame

    Notes
    -----
    Requires a minimum of two columns: a target column and a data column. There must
    always be 1 target column, and there can be N>0 columns for data.
    """
    if not columns:
        columns = list(window)

    if target in columns:
        columns.remove(target)

    columns += ['windowID', 'datetime']
    df = pd.DataFrame({col: window[col] for col in columns})

    extracted = tsfresh.extract_features(
        df,
        column_id    = 'windowID',
        column_sort  = 'datetime',
        column_kind  = None,
        column_value = None,
        default_fc_parameters = features,
        disable_progressbar   = True,
        n_jobs = 1
    )

    # Add the excluded columns back in
    excluded = list(set(window) - set(columns)) + ['datetime']
    for col in excluded:
        extracted[col] = window[col][index]

    # If this is a classification problem, modify the target
    if classification and window[target].any():
        extracted[target] = 1

    return extracted


def rotate(window, columns=[], index=-1, target=None, classification=False, **kwargs):
    """
    Rotates a DataFrame by stacking columns and converting to a single row DataFrame.
    Sets the value of the last index as the index of the rotated DataFrame.

    Parameters
    ----------
    window : dict
        Dict in format {column: np.array}
    columns : list, default=[]
        Columns to rotate, all others will be re-added as the value from the specified index
    index : int, default=-1
        Index set the window at. Defaults to -1 which is the last index of the window.
    target : str, default=None
        This is the target variable
    classification : bool, default=False
        If true, changes the target to 1 if any in the window
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
    >>> Logger.info(rotated_df)
        A_0  A_1  A_2  B_0  B_1  B_2  C_0  C_1  C_2
    -3    1    2    3    4    5    6    7    8    9
    """
    if not columns:
        columns = list(window)

    if target in columns:
        columns.remove(target)

    rotated = {}
    for col in columns:
        for i, val in enumerate(window[col]):
            rotated[f'{col}_{i}'] = [val]

    excluded = set(window) - set(columns)
    for col in excluded:
        rotated[col] = [window[col][index]]

    if classification and window[target].any():
        rotated[target] = [1]

    return rotated


def findDatasets(h5, keys):
    """
    Finds all key paths in an H5 file containing h5py._hl.dataset.Dataset objects
    """
    if not isinstance(keys, list):
        keys = [keys]

    if isinstance(h5, str):
        with h5py.File(h5, 'r') as h5:
            return findDatasets(h5, keys)

    valid = []
    for key in keys:
        if isinstance(group := h5[key], h5py._hl.group.Group):
            subkeys = [f'{key}/{sub}' for sub in group.keys()]
            sub = findDatasets(h5, subkeys)

            # This key has Dataset objects, append it
            if sub is True:
                valid.append(key)
            else:
                # A list of valid keys was returned
                valid += sub
        else:
            # This was a Dataset, return True that the parent is valid
            return True

    return valid


class Extract:
    def __init__(self):
        """
        """
        self.C = Config.extract
        self.model_type   = Config.model.type
        self.model_target = Config.model.target or None

        if self.C.ray:
            ray.init(**self.C.ray)

        # Determine what keys from the H5 file to load
        match (keys := self.C.multi):
            # Retrieve all subkeys from this group
            case str() | list():
                keys = findDatasets(Config.extract.file, keys)
                self.multi = True
                self.metadata = {}

            # Single track case
            case _:
                keys = ['preprocess/complete']
                self.multi = False
                self.metadata = None

        # Select a window processing method
        params = {
            'target'        : self.model_target,
            'columns'       : self.C.get('columns', []),
            'index'         : self.C.get('index'  , -1),
            'classification': self.model_type == 'classification',
        }
        match self.C.method:
            case "tsfresh":
                process = extract
                params['features'] = get_features(**self.C.features)
            case "rotate":
                process = rotate
            case "passthrough":
                process = passthrough
            case invalid:
                raise AttributeError(f"Invalid method chosen: {invalid}")

        # Perform the processing via Ray
        Logger.info('Beginning processing')
        for key in tqdm(keys, position=1, desc='Processing Frames'):
            try:
                self.process(key, process, params)
            except:
                Logger.exception(f'Failed to process key: {key}')

        # Save the metadata, if there was any
        if self.multi:
            if (file := Config.subselect.metadata):
                if self.metadata:
                    Logger.info(f'Saving metadata to: {file}')
                    utils.save_pkl(file, self.metadata)
                else:
                    Logger.error('No metadata produced for a multi-track case, this may have consequences down the line')
            else:
                Logger.error('Classification runs must define Config.subselect.metadata')


    def loadAndRoll(self, key):
        """
        """
        Logger.info(f'Loading key {key!r} from {self.C.file}')
        df = pd.read_hdf(self.C.file, key)

        if not isinstance(df.index, pd.DatetimeIndex):
            Logger.warning(f'The index is not a DatetimeIndex for key {key}')

            datetimes = [col for col, data in df.items() if pd.api.types.is_datetime64_any_dtype(data.dtype)]
            if datetimes:
                Logger.warning(f'The following columns were detected to be datetime dtypes, using the first available: {datetimes}')
                df = df.set_index(datetimes[0])
            else:
                Logger.error('No datetime column found, cannot proceed')
                return

        Logger.info('Calculating windows')
        windows = Roll(df, method='groups', **self.C.roll)

        Logger.info(f'Valid windows: {windows.roll()}')

        return windows


    def process(self, key, func, params):
        """
        """
        windows = self.loadAndRoll(key)

        if windows is None:
            return

        if windows.valid == 0:
            Logger.error('No windows were accepted for this track of data. Nothing to do, returning nothing')
            return

        # Convert the index back to a column for processing, will set back later
        index = windows.windows.index.name or 'index'
        data  = windows.windows.reset_index()

        # Find the ideal number of blocks to split such that the expected number of extracted windows is produced
        # Start with a minimum number of blocks as 20
        samples   = data.shape[0]
        total     = samples / windows.size
        blocks    = self.C.blocks.min
        maxBlocks = self.C.blocks.max

        if isinstance(maxBlocks, float):
            maxBlocks *= total

        for blocks in range(blocks, int(maxBlocks)):
            if total % blocks == 0:
                Logger.debug(f'Blocks: {blocks}')
                break
        else:
            Logger.debug(f'Could not calculate an ideal block size')
            blocks = None

        # Cast to ray.data
        ds = ray.data.from_pandas(data)
        if blocks:
            ds = ds.materialize().repartition(blocks)

        Logger.info('Starting processes')
        start = dtt.now()

        ds = ds.map_batches(func,
            batch_size      = windows.size,
            concurrency     = os.cpu_count(),
            zero_copy_batch = True,
            fn_kwargs       = params
        )
        file = f'{self.C.parquet}/{key}'
        Logger.info(f'Writing to parquet: {file}')
        ds.write_parquet(file)

        # Release resources
        del ds, windows, data

        Logger.info(f'Elapsed time: {dtt.now() - start}')

        self.loadAndFinish(key, index)


    def loadAndFinish(self, key, index):
        """
        """
        ds = ray.data.read_parquet(f'{self.C.parquet}/{key}')
        df = ds.to_pandas()
        df = df.set_index(index)
        df = verify(df)

        if self.multi:
            self.extract_metadata(df, key)
        else:
            Logger.info('Saving to key extract/complete')
            df.to_hdf(self.C.file, key=f'extract/complete')


    def extract_metadata(self, df, key):
        """
        """
        key = f'extract/tracks/{key}'

        counts = {}
        if self.model_type == 'classification':
            counts = df[self.model_target].value_counts()

        self.metadata[key] = {
            'start': df.index[0],
            'end'  : df.index[-1],
            'neg'  : counts.get(0, 0),
            'pos'  : counts.get(1, 0)
        }

        Logger.debug(f'Saving to key {key}')
        df.to_hdf(self.C.file, key=key)


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
