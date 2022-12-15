import argparse
import h5py
import logging
import numpy  as np
import pandas as pd
import re
import tsfresh

from tqdm                 import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from tsfresh.feature_extraction import ComprehensiveFCParameters

from timefed        import utils
from timefed.config import (
    Config,
    Section
)

Logger = logging.getLogger('timefed/extract.py')

def report(stats, print=print):
    """
    """
    print('Roll stats:')
    print(f'- Frequency of the data is: {stats.frequency}')
    print(f'- The data ranges over {stats.range}')
    print(f'- Using a window size of {stats.window} and a step of {stats.step}, the size of each window is {stats.size} samples')
    print(f'- Windows produced:')
    print(f'-- Total possible : {stats.possible}')
    print(f'-- Number accepted: {stats.valid} ({stats.valid/stats.possible*100:.2f}%)')

    if stats.optional:
        print(f'-- Number of windows containing each optional variable:')
        utils.align_print(stats.optional, print=print, prepend='--- ')

    if stats.possible != stats.valid:
        print(f'-- Number rejected: {stats.possible-stats.valid} ({(stats.possible-stats.valid)/stats.possible*100:.2f}%)')
        print(f'-- Reasons for rejection:')
        utils.align_print(stats.reasons, print=print, prepend='--- ')

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
    stats = Section('roll stats', {
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
        Logger.warning('Duplicate timestamps were detected, windowing my return unexpected results')

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

def get_features(whitelist=None, blacklist=None, prompt=False):
    """
    Retrieves the dictionary of tsfresh features to use and their arguments.
    Can be used in an interactive mode or configured via the configuration file.

    Parameters
    ----------
    whitelist : list
        List of feature names to only include for calculations
    blacklist : list
        List of feature names to exclude from calculations
    prompt : bool
        Enables interactive mode for this function, printing the features list
        to screen and prompting for a list to use for calculations

    Returns
    -------
    features : dictionary
        Dictionary of {feature_name: feature_arguments} for tsfresh to use
        during feature extraction
    """
    features = ComprehensiveFCParameters()

    # Apply white/black lists if available
    if whitelist:
        features = {key: value for key, value in features.items() if key in whitelist}
    if blacklist:
        features = {key: value for key, value in features.items() if key not in blacklist}

    # Prompt the user with a list of available features
    if prompt:
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
    Verifies
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
    pd.options.mode.chained_assignment = None

    if slice:
        df = df.iloc[slice, :]

    if not columns:
        columns = list(df.columns)

    columns += ['_ID', '_TIME']

    if target in columns:
        columns.remove(target)

    df['_ID']   = np.full(len(df), 0)
    df['_TIME'] = df.index
    extracted   = tsfresh.extract_features(
        df[columns],
        column_id    = '_ID',
        column_sort  = '_TIME',
        column_kind  = None,
        column_value = None,
        default_fc_parameters = features,
        disable_progressbar   = True,
        n_jobs = 1
    )

    # Imitate the original index
    extracted.index = [df.index[index]]

    # Add the excluded columns back in
    excluded = list(set(df.columns) - set(columns))
    if excluded:
        extracted[excluded] = df[excluded].iloc[index]

    # If this is a classification problem, modify the target
    if classification and df[target].any():
        extracted[target] = 1

    return extracted

def process(df, features=None):
    """
    Uses Ray as a multiprocessing backend to quickly process
    many windows through extract. This is an optional function
    and may be used as a guideline to writing custom scripts
    leveraging the TimeFED API.

    Parameters
    ----------
    df: pandas.core.DataFrame
        A single input of data to create windows on and process through extraction
    features: dict, default=None
        Dictionary subset of tsfresh.feature_extraction.ComprehensiveFCParameters
        Defaults to None which uses the default ComprehensiveFCParameters (all features, default parameters)

    Returns
    -------
    bool
        Whether the function finished correctly
    """
    import ray

    config = Config()

    # Only initialize if it's provided by the config
    if config.ray:
        ray.init(**config.ray)

    windows, stats = roll(df,
        window     = config.window,
        frequency  = config.frequency,
        step       = config.step or 1,
        required   = config.required,
        optional   = config.optional,
        as_frames  = False
    )

    report(stats, Logger.info)

    # Place constant params into shared memory
    params = {
        'df'            : ray.put(df),
        'features'      : ray.put(features),
        'columns'       : ray.put(config.get('columns'       , []     )),
        'target'        : ray.put(config.get('target'        , None   )),
        'index'         : ray.put(config.get('index'         , -1     )),
        'classification': ray.put(config.get('classification', False  )),
    }
    func = ray.remote(extract)
    jobs = [func.remote(**params, slice=slice(*window)) for window in windows]

    extracts = []
    for i in tqdm(range(len(windows)), desc='Processing Windows', position=0):
        [done], running = ray.wait(jobs, num_returns=1)
        jobs   = running
        window = ray.get(done)

        if config.output.windows:
            window.to_hdf(config.output.windows, f'window_{i}')
        else:
            extracts.append(window)

        del window, done

    if config.output.windows:
        Logger.info('Loading windows into memory')
        extracts = [pd.read_hdf(config.output.windows, f'window_{j}') for j in range(i+1)]

    Logger.info('Concatting the feature frames together')
    df = pd.concat(extracts).sort_index()
    df = verify(df)

    return df

@utils.timeit
def main():
    """
    The main process of TrackWindow
    """
    # Retrieve the config object
    config = Config()

    # Retrieve features to use
    features = get_features(
        whitelist = config.features.whitelist,
        blacklist = config.features.blacklist,
        prompt    = config.features.interactive
    )

    if config.input.multi:
        if isinstance(config.input.key, str):
            with h5py.File(config.input.file, 'r') as h5:
                config.input.key = [
                    f'{config.input.key}/{key}'
                    for key in h5[config.input.key].keys()
                ]

        metadata = {}
        for key in tqdm(config.input.key, position=1, desc='Streams Processed'):
            df = pd.read_hdf(config.input.file, key)
            df = process(df, features)

            # Metadata information is used by subselect.py for the multitrack case
            counts = {}
            if config.classification:
                counts = df[config.target].value_counts()

            metadata[key] = {
                'start': df.index[0],
                'end'  : df.index[-1],
                'neg'  : counts.get(0, 0),
                'pos'  : counts.get(1, 0)
            }

            df.to_hdf(config.output.file, f'windows/{key}')

        if config.classification:
            utils.save_pkl(config.output.metadata, metadata)

    else:
        df = pd.read_hdf(config.input.file, config.input.key)
        df = process(df, features)
        df.to_hdf(config.output.file, 'windows')

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/config.yaml',
                                            help     = 'Path to a config.yaml file'
    )
    parser.add_argument('-s', '--section',  type     = str,
                                            default  = 'extract',
                                            help     = 'Section of the config to use'
    )

    args = parser.parse_args()

    try:
        utils.init(args)

        with logging_redirect_tqdm():
            code = main()

        Logger.info('Finished successfully')
    except Exception as e:
        Logger.exception('Failed to complete')
