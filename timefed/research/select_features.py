import argparse
import logging
import pandas as pd

from tsfresh import select_features

from timefed import utils
from timefed.config import Config

Logger = logging.getLogger('timefed/select.py')

def split(df):
    """
    """
    config = Config().split

    # Make sure date string doesn't use -
    config.train = config.train.replace('-', '_')
    config.test  = config.test .replace('-', '_')

    Logger.debug(f'Train query: "index {config.train}"')
    Logger.debug(f'Test  query: "index {config.test}"')

    train = df.query(f'index {config.train}')
    test  = df.query(f'index {config.test}')

    Logger.debug(f'Train shape: {train.shape} ({train.shape[0]/df.shape[0]*100:.2f}%)')
    Logger.debug(f'Test  shape: {test.shape} ({test.shape[0]/df.shape[0]*100:.2f}%)')

    return train, test

def select():
    """
    Selects relevant features from a tsfresh dataframe of features for a target
    label
    """
    config = Config()

    Logger.info(f'Loading data from {config.input.file} using key {config.input.key}')
    df = pd.read_hdf(config.input.file, config.input.key)

    if config.drop:
        Logger.info(f'Dropping columns: {config.drop}')
        df = df.drop(columns=config.drop)

    Logger.info('Splitting data into train/test sets')
    train, test = split(df)

    # Select only on the train data
    label = train[config.label]

    Logger.info('Selecting features on the train set')
    train = select_features(train.drop(columns=[config.label]), label, n_jobs=config.get('cores', 1), chunksize=64)

    # Add the label column back in
    train[config.label] = label

    # Only keep the same features in test as train
    test = test[train.columns]

    # Report stats
    Logger.info(f'Number of selected features: {train.shape[1]-1}/{df.shape[1]-1} ({(train.shape[1]-1)/(df.shape[1]-1)*100:.2f})')

    Logger.info(f'Saving to {config.output.file} under key {config.output.key}/[train,test]')
    train.to_hdf(config.output.file, f'{config.output.key}/train')
    test .to_hdf(config.output.file, f'{config.output.key}/test')

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/config.yaml',
                                            help     = 'Path to a config.yaml file'
    )
    parser.add_argument('-s', '--section',  type     = str,
                                            default  = 'select',
                                            help     = 'Section of the config to use'
    )

    args = parser.parse_args()

    # Config arguments required for the script
    args.require = {
        'input': {
            'file': str,
            'key': str
        },
        'output': {
            'file': str,
            'key': str
        },
        'split': {
            'train': str,
            'test': str
        },
        'label': str
    }

    try:
        utils.init(args)

        code = select()

        Logger.info('Finished successfully')
    except Exception as e:
        Logger.exception('Failed to complete')
