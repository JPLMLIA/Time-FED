import argparse
import logging
import numpy  as np
import pandas as pd
import re
import tsfresh
import warnings

from functools import partial
from tables    import NaturalNameWarning
from tqdm      import tqdm

from tsfresh.feature_extraction import ComprehensiveFCParameters

from timefed import utils
from timefed.config import Config

# Disable h5py warning about setting an integer as a key name
warnings.filterwarnings('ignore', category=NaturalNameWarning)


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
    step    = pd.Timedelta(step)
    size    = df.shape[0] - 1
    windows = []

    # Setup the progress bar
    perc = np.linspace(1, size - observations - 1, 100)
    bar  = tqdm(total=100, desc='Percent Rolled')
    prog = 0

    i = -1
    while i < size - observations:
        i += 1
        j  = i + observations
        if index[j] - index[i] > delta:
            continue

        windows.append(df.iloc[i:j])

        # Incrementally step the progress bar
        if (i > perc).sum() > prog:
            prog += 1
            bar.update()

    return windows

def roll_v1(df, window, step, observations):
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
    # Setup the progress bar
    bar  = tqdm(total=100, desc='Percent Rolled')
    prog = 0

    index   = df.index
    delta   = pd.Timedelta(window)
    step    = pd.Timedelta(step)
    size    = df.shape[0]
    windows = []

    i = 0
    while i < size:
        # Find the next index
        j = i + 1
        while j < size and index[j] - index[i] < delta:
            j += 1

        sub = df.iloc[i:j]

        # Make sure the window is the correct size
        if sub.shape[0] == observations:
            windows.append(sub)

            # Make sure the step is correct
            k = i + 1
            while k < size and index[k] - index[i] < step:
                k += 1
            i = k
        else:
            # Step only one index if this window was invalid
            i += 1

        # Incrementally step the progress bar
        p = np.round(i / size, decimals=2)
        while p > prog:
            bar.update()
            prog += 0.01

    return windows

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

def extract(df, config, features=None):
    """
    """
    df['_ID']   = np.full(len(df), 0)
    df['_TIME'] = df.index
    extracted   = tsfresh.extract_features(
        df.drop(columns=config.drop),
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
    extracted[config.drop] = df[config.drop].iloc[-1]

    # If this is a classification problem, modify the label
    if config.kind == 'classification':
        if df[config.label].any():
            extracted[config.label] = 1

    return extracted

@utils.timeit
def process():
    """
    The main process of TrackWindow
    """
    # Retrieve the config object
    config = Config()

    # Retrieve features to use
    features = get_features(
        whitelist = config.features.whitelist,
        blacklist = config.features.blacklist,
        prompt    = config.prompt.features
    )
    func = partial(extract, features=features, config=config)

    # load the data
    df = pd.read_hdf(config.input.file, config.input.key)

    # Make sure there are no NaNs
    if df.isna().any().any():
        Logger.error('There are NaNs in the DataFrame. Cannot continue processing. Please check the debug log for more detail.')
        Logger.debug(f'Percent of NaNs in each column:\n{(df.isna().sum() / df.index.size) * 100}')
        return 1

    # Determine which columns to use for processing
    if config.tsfresh:
        config.drop = [config.label] + list(set(df.columns) - set(config.tsfresh))

    # The label column is always excluded from feature extraction
    if not config.drop:
        config.drop = [config.label]

    # Remove columns that are not int or float dtypes
    for col in df:
        dtype = df[col].dtype
        if not np.issubdtype(dtype, int) and not np.issubdtype(dtype, float):
            config.drop.append(col)

    # Unique the drops list
    config.drop = list(set(config.drop))
    Logger.debug(f'tsfresh will extract only on the following columns: {set(df.columns) - set(config.drop)}')

    Logger.info('Creating the rolling windows and beginning processing')
    # Create rolling windows and process tsfresh on each window
    windows = roll(df,
        window       = config.window,
        step         = config.step,
        observations = config.observations
    )
    Logger.info(f'Number of windows: {len(windows)}')

    bar = tqdm(total=len(windows), desc='Extracting')
    with utils.Pool(processes=config.cores) as pool:
        i = 0
        for ret in pool.imap_unordered(func, windows, chunksize=100):
            # ret.to_hdf(config.output.file, f'{config.output.key}/windows/{i}')
            bar.update()

    Logger.info('Concatting the feature frames together')
    ret = pd.concat(extracts)
    ret.sort_index(inplace=True)

    Logger.info(f'Saving to {config.output.file}')
    ret.to_hdf(config.output.file, f'{config.output.key}/full')

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/config.yaml',
                                            help     = 'Path to a config.yaml file'
    )
    parser.add_argument('-s', '--section',  type     = str,
                                            default  = 'extract_features',
                                            help     = 'Section of the config to use'
    )

    args = parser.parse_args()

    try:
        config = Config(args.config, args.section)

        utils.init(config)
        Logger = logging.getLogger('timefed/extract.py')

        code = process()

        Logger.info('Finished successfully')
    except Exception as e:
        Logger.exception('Failed to complete')
