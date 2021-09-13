import argparse
import logging
import pandas as pd
import numpy  as np
import os

from pvlib.solarposition import get_solarposition

# Import utils to set the logger
from mloc import utils

logger = logging.getLogger('mloc/research/preprocess.py')

def filter(feature, args, df):
    """
    Filters a feature's values given a set of arguments

    Parameters
    ----------
    args : utils.Config
        Config object defining arguments for filtering
    df : pandas.DataFrame
        The dataframe to filter on

    Returns
    -------
    df : pandas.DataFrame
        The filtered dataframe
    """
    old_count = (~df[feature].isna()).sum()

    if 'lt' in args:
        logger.debug(f'\t< {args.lt}')
        df = df[df[feature] < args.lt]

    if 'gt' in args:
        logger.debug(f'\t> {args.gt}')
        df = df[df[feature] > args.gt]

    if 'lte' in args:
        logger.debug(f'\t<= {args.lte}')
        df = df[df[feature] <= args.lte]

    if 'gte' in args:
        logger.debug(f'\t>= {args.gte}')
        df = df[df[feature] >= args.gte]

    new_count = (~df[feature].isna()).sum()
    logger.debug(f'\tFiltered {(1-new_count/old_count)*100:.2f}% of the data')

    return df

def calculate_features(df, config):
    """
    Calculates new features given a config

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to calculate new features on
    config : utils.Config
        Configuration containing which features to calculate
    """
    # Always override the SZA feature (or add it)
    df['solar_zenith_angle'] = get_solarposition(
        time      = df.index,
        latitude  = 34.380000000000003,
        longitude = -1.176800000000000e+02,
        altitude  = 2280
    ).zenith

    if 'month' in config.calc:
        df['month'] = df.index.month

    if 'day' in config.calc:
        df['day']   = df.index.dayofyear

    if 'minute' in config.calc:
        df['minute'] = df.index.hour * 60 + df.index.minute

    if 'reitan' in config.calc:
        df['reitan'] = df['drew_drop'] * 2 # dummy calc

    for feature in config.log:
        if feature in df:
            df[f'log_{feature}'] = np.log10(df[feature])
        else:
            logger.error(f'Feature not found for log: {feature}')

    return df

def preprocess(config):
    """
    Preprocesses the dataframe with additional features

    Parameters
    ----------
    config : utils.Config
        Config object containing arguments for preprocessing
    """
    # Load the data dataframe
    df = pd.read_hdf(config.input.file, config.input.key)

    if config.subselect:
        logger.info('Subselecting whole dataset')
        df = utils.subselect(config.subselect, df).copy()

    logger.debug(f'df.describe():\n{df.describe()}')

    logger.info('Creating new features')
    df = calculate_features(df, config.features)

    # Apply filtering
    if config.filter:
        for feature, args in vars(config.filter).items():
            logger.debug(f'Filtering {feature}')
            df = filter(feature, args, df)

    logger.debug(f'Count of non-NaN values:\n{(~df.isnull()).sum()}')

    if config.exclude:
        cols = set(df.columns)
        drop = cols - (cols - set(config.exclude))
        df   = df.drop(columns=drop)

    if config.include:
        cols = set(df.columns)
        drop = cols - set(config.include)
        df   = df.drop(columns=drop)

    # Write to output
    df.to_hdf(config.output.file, f'{config.output.key}/full')

    if config.train:
        logger.info('Creating training subset')
        train = utils.subselect(config.train, df)
        logger.debug(f'Count of non-NaN values for train:\n{(~train.isnull()).sum()}')
        train.to_hdf(config.output.file, f'{config.output.key}/train')
        # Interpolate train only

    if config.test:
        logger.info('Creating testing subset')
        test = utils.subselect(config.test, df)
        logger.debug(f'Count of non-NaN values for test:\n{(~test.isnull()).sum()}')
        test.to_hdf(config.output.file, f'{config.output.key}/test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/config.yaml',
                                            help     = 'Path to a config.yaml file'
    )
    parser.add_argument('-s', '--section',  type     = str,
                                            default  = 'preprocess',
                                            help     = 'Section of the config to use'
    )

    args = parser.parse_args()

    try:
        config = utils.Config(args.config, args.section)

        preprocess(config)

        logger.info('Finished successfully')
    except Exception:
        logger.exception('Failed to complete')
