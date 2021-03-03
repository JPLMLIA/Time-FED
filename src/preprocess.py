import argparse
import logging
import pandas as pd
import numpy  as np
import os

# Import utils to set the logger
import utils

logger = logging.getLogger(os.path.basename(__file__))

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
        logger.debug(f'\t< {args.lt}')
        sub = sub[sub.index < args.lt]

    if 'gt' in args:
        logger.debug(f'\t> {args.gt}')
        sub = sub[sub.index > args.gt]

    if 'lte' in args:
        logger.debug(f'\t<= {args.lte}')
        sub = sub[sub.index <= args.lte]

    if 'gte' in args:
        logger.debug(f'\t>= {args.gte}')
        sub = sub[sub.index >= args.gte]

    return sub

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
        df = subselect(config.subselect, df).copy()

    logger.debug(f'df.describe():\n{df.describe()}')

    logger.info('Creating new features')
    if 'month' in config.features:
        df['month'] = df.index.month
    if 'day' in config.features:
        df['day']   = df.index.dayofyear
    if 'logCn2' in config.features:
        df['logCn2'] = np.log10(df['Cn2'])

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

    # Write to output
    df.to_hdf(config.output.file, f'{config.output.key}/full')

    if config.train:
        logger.info('Creating training subset')
        train = subselect(config.train, df)
        logger.debug(f'Count of non-NaN values for train:\n{(~train.isnull()).sum()}')
        train.to_hdf(config.output.file, f'{config.output.key}/train')
        # Interpolate train only

    if config.test:
        logger.info('Creating testing subset')
        test = subselect(config.test, df)
        logger.debug(f'Count of non-NaN values for test:\n{(~test.isnull()).sum()}')
        test.to_hdf(config.output.file, f'{config.output.key}/test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/config.yaml',
                                            help     = 'Path to a config.yaml file'
    )

    args = parser.parse_args()

    try:
        config = utils.Config(args.config, 'preprocess')

        preprocess(config)

        logger.info('Finished successfully')
    except Exception:
        logger.exception('Failed to complete')
