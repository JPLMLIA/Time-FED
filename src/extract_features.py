import argparse
import logging
import numpy  as np
import pandas as pd
import re

from functools import partial
from tqdm      import tqdm

from tsfresh import (
    extract_features,
    select_features
)
from tsfresh.feature_extraction            import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute

import utils

logger = logging.getLogger('mloc/extract_features.py')

def roll(df, window, step=1, observations=None, drop=None):
    """
    Creates a generator for rolling over a pandas DataFrame with a given window
    size.

    Parameters
    ----------
    df : pandas.DataFrame
    window : str or int
        The window size to extract
    step : int
        Step size to take when rolling over the DataFrame
    observations : int
        Minimum number of observations required to be a valid window
    drop : list
        List of variables to drop before yielding. This allows for variables to
        be considered in the observations during rolling but excluded from
        tsfresh extractions.

    Yields
    ------
    pandas.DataFrame
    """
    size = df.index.size
    if isinstance(window, str):
        delta = pd.Timedelta(window)
        for i in tqdm(range(0, size, step), desc='Rolling'):
            if i < size:
                j = i+1
                while j < size and (df.index[j] - df.index[i]) < delta:
                    j += 1

                sub = df.iloc[i:j]

                if observations:
                    if sub.index.size < observations:
                        continue

                if drop:
                    sub = sub.drop(columns=drop)

                yield sub

    elif isinstance(window, int):
        for i in tqdm(range(0, size, step), desc='Rolling'):
            if i < size:
                sub = df.iloc[i:min(i+window, size)]
                if drop:
                    sub = sub.drop(columns=drop)
                yield sub

def get_features(whitelist=None, blacklist=None, prompt=False):
    """
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
        logger.info('Current list of features to be used in feature extraction:')

        for i, feat in enumerate(retain):
            logger.info(f'\t{i}\t- {feat}')

        response = input('Please select which features to use in extraction (eg. 0 3 11): ')
        indices  = re.findall(r'(\d+)', response)
        retain   = [retain[int(i)] for i in indices]

        features = {key: value for key, value in features.items() if key in retain}

    return features

def extract(df, features=None):
    """
    """
    columns     = df.columns
    df['_ID']   = np.full(len(df), 0)
    df['_TIME'] = df.index
    extract = extract_features(
        df,
        column_id    = '_ID',
        column_sort  = '_TIME',
        column_kind  = None,
        column_value = None,
        impute_function       = impute,
        default_fc_parameters = features,
        disable_progressbar   = True,
        n_jobs = 1
    )

    # Imitate the original index
    extract.index = [df.index[-1]]

    return extract

def median(df):
    """
    Extracts the median value from the window as the value
    """
    nf = df.median().to_frame().T
    nf.index = [df.index[-1]]

    return nf

@utils.timeit
def process(config):
    """
    The main process of TrackWindow
    """
    # Retrieve features to use
    features = get_features(
        whitelist = config.features.whitelist,
        blacklist = config.features.blacklist,
        prompt    = config.prompt.features
    )
    func = partial(extract, features=features)

    # load the data and drop nan values
    df   = pd.read_hdf(config.input.file, config.input.key)
    orig = df.index.size
    df   = df.dropna(how='any', axis=0)
    logger.debug(f'Dropping NaNs reduced the data by {(1-df.index.size/orig)*100:.2f}%')

    logger.info('Creating the rolling windows and beginning processing')
    # Create rolling windows and process tsfresh on each window
    rolls = roll(df,
        window       = config.window,
        step         = config.step,
        observations = config.observations,
        drop         = config.drop
    )
    extracts = []
    with utils.Pool(processes=config.cores) as pool:
        if config.process == 'tsfresh':
            for ret in pool.imap_unordered(func, rolls, chunksize=200):
                extracts.append(ret)
        elif config.process == 'median':
            for ret in pool.imap_unordered(median, rolls, chunksize=200)

    logger.info('Concatting the feature frames together')
    ret = pd.concat(extracts)
    ret.sort_index(inplace=True)

    # Add back in the original columns
    ret[df.columns] = df.loc[ret.index]

    if config.output.file:
        logger.info(f'Saving raw to {config.output.file}')
        ret.to_hdf(config.output.file, f'{config.output.key}/full')

    # Select features
    if config.process == 'tsfresh':
        logger.info('Selecting relevant features')
        label = ret[config.label]
        ret   = select_features(ret.drop(columns=[config.label]), label, n_jobs=config.cores, chunksize=100)

        # Add the label column back in
        ret[config.label] = label

        if config.output.file:
            logger.info(f'Saving to {config.output.file}')
            ret.to_hdf(config.output.file, f'{config.output.key}/select')

    return ret


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
        config = utils.Config(args.config, args.section)

        process(config)

        logger.info('Finished successfully')
    except Exception as e:
        logger.exception('Failed to complete')
