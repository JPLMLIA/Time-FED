import argparse
import logging
import pandas as pd

from mlky    import Config
from tsfresh import select_features

from timefed.utils import utils

Logger = logging.getLogger('timefed/core/selfeatures.py')


def split(df):
    """
    """
    config = Config.selfeatures.split

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
    """
    Logger.info(f'Loading data from {Config.selfeatures.file}')
    df = pd.read_hdf(Config.selfeatures.file, 'extract/complete')

    if Config.selfeatures.drop:
        Logger.info(f'Dropping columns: {Config.selfeatures.drop}')
        df = df.drop(columns=Config.selfeatures.drop)

    Logger.info('Splitting data into train/test sets')
    train, test = split(df)

    # Perform feature selection only on the train data, excluding target
    target = train[Config.model.target]

    Logger.info('Selecting features on the train set')
    train = select_features(train.drop(columns=[Config.model.target]), target, n_jobs=Config.selfeatures.get('cores', 1), chunksize=64)

    # Add the target column back in
    train[Config.model.target] = target

    # Only keep the same features in test as train
    test = test[train.columns]

    # Report stats
    Logger.info(f'Number of selected features: {train.shape[1]-1}/{df.shape[1]-1} ({(train.shape[1]-1)/(df.shape[1]-1)*100:.2f})')

    Logger.info(f'Saving to {Config.selfeatures.file} under key selfeatures/[train,test]')
    train.to_hdf(Config.selfeatures.file, f'selfeatures/train')
    test .to_hdf(Config.selfeatures.file, f'selfeatures/test')

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--Config.selfeatures',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/Config.selfeatures.yaml',
                                            help     = 'Path to a Config.selfeatures.yaml file'
    )
    parser.add_argument('-s', '--section',  type     = str,
                                            default  = 'select',
                                            help     = 'Section of the Config.selfeatures to use'
    )

    args = parser.parse_args()

    try:
        utils.init(args)

        code = select()

        Logger.info('Finished successfully')
    except Exception as e:
        Logger.exception('Failed to complete')
