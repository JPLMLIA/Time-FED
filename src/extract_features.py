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

def select(df, config, shift=None):
    """
    Selects relevant features from a tsfresh dataframe of features for a target
    label

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing extracted features from tsfresh
    config : utils.Config
        Configuration object
    """
    # Select only on the train subset
    train = utils.subselect(config.train, df)
    test  = utils.subselect(config.test,  df)

    # Select features
    label = train[config.label]
    train = select_features(train.drop(columns=[config.label]), label, n_jobs=config.cores, chunksize=100)

    # Add the label column back in
    train[config.label] = label

    # Only keep the same features in test as train
    test = test[train.columns]

    if config.output.file:
        logger.info(f'Saving to {config.output.file}')
        if shift:
            train.to_hdf(config.output.file, f'{config.output.key}/train/historical_{shift}_min')
            test.to_hdf(config.output.file, f'{config.output.key}/test/historical_{shift}_min')
        else:
            train.to_hdf(config.output.file, f'{config.output.key}/train')
            test.to_hdf(config.output.file, f'{config.output.key}/test')

@utils.timeit
def process(config):
    """
    The main process of TrackWindow
    """
    if config.process == 'tsfresh':
        # Retrieve features to use
        features = get_features(
            whitelist = config.features.whitelist,
            blacklist = config.features.blacklist,
            prompt    = config.prompt.features
        )
        func = partial(extract, features=features)
    elif config.process == 'median':
        func = median
    else:
        logger.error(f'Unrecognized process argument: {config.process}')
        return

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
        for ret in pool.imap_unordered(func, rolls, chunksize=200):
            extracts.append(ret)

    logger.info('Concatting the feature frames together')
    ret = pd.concat(extracts)
    ret.sort_index(inplace=True)

    # Add back in the original columns
    if config.process == 'tsfresh':
        ret[df.columns] = df.loc[ret.index]
    elif config.process == 'median':
        columns = list(set(df.columns) - set(ret.columns))
        ret[columns] = df[columns].loc[ret.index]

    if config.output.file:
        logger.info(f'Saving raw to {config.output.file}')
        ret.to_hdf(config.output.file, f'{config.output.key}/full')

    # Select features
    if config.process == 'tsfresh':
        if config.label in ret:
            for length in config.historical:
                logger.info(f'Selecting relevant features for historical length {length} minutes')
                ## Shift by the historical length
                # Copy the original
                shift = ret.copy()
                label = ret[config.label]

                # Create a copy of the label column and drop the label
                shift[f'{config.label}_H{length}'] = label
                shift = shift.drop(columns=[label])

                # Shift the index by the length amount in minutes, add label back in
                shift.index += pd.Timedelta('{length} min')
                shift[config.label] = label

                select(shift, config, shift=length)
        else:
            logger.info('Selecting relevant features')
            select(ret, config)

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
        config = utils.Config(args.config, args.section)

        process(config)

        logger.info('Finished successfully')
    except Exception as e:
        logger.exception('Failed to complete')
