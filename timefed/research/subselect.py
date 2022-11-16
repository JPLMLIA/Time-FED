import argparse
import logging
import numpy  as np
import pandas as pd

from tqdm    import tqdm
from tsfresh import select_features

from timefed        import utils
from timefed.config import Config

Logger = logging.getLogger('timefed/subselect.py')


def split(df, date):
    """
    Parameters
    ----------
    df: pandas.core.DataFrame
    date:

    Returns
    -------
    train: pandas.core.DataFrame
    test: pandas.core.DataFrame
    """
    Logger.debug(f'Split date selected: {date}')

    train = df.query('index <  @date')
    test  = df.query('index >= @date')

    Logger.debug(f'Train shape: {train.shape} ({train.shape[0]/df.shape[0]*100:.2f}%)')
    Logger.debug(f'Test  shape: {test.shape} ({test.shape[0]/df.shape[0]*100:.2f}%)')

    return train, test

def select(df, train, test, target='label', n_jobs=1):
    """
    Selects relevant features from a tsfresh dataframe of features for a target
    label

    Parameters
    ----------

    Returns
    -------
    """
    # Select only on the train data
    label = train[target]

    Logger.info('Selecting features on the train set')
    train = select_features(train.drop(columns=[target]), label,
        n_jobs    = n_jobs,
        chunksize = 2**10
    )

    # Add the target column back in
    train[target] = label

    # Only keep the same features in test as train
    test = test[train.columns]

    Logger.info(f'Number of selected features: {train.shape[1]-1}/{df.shape[1]-1} ({(train.shape[1]-1)/(df.shape[1]-1)*100:.2f})')

    return train, test

def _split_single_classification(df, n=10, target='label'):
    """
    Parameters
    ----------
    df: pandas.core.DataFrame
        DataFrame to split
    n: int
        Number of splits to perform where step is 100/n
    target: str
        The column to use as the feature. Values are expected to be binary

    Returns
    -------
    """
    # [S]plit [F]rame, DataFrame to store possible splits
    sf = pd.DataFrame(columns=pd.MultiIndex.from_product([['Train', 'Test'], ['Total', 'Percent'], [0, 1]]), index=range(n-1))
    sf['Train% / Test%'] = np.nan
    sf['Split Date']     = np.nan

    total  = df.shape[0]
    labels = df[target].value_counts()
    groups = [int(i*total/n) for i in range(1, n)]
    for i, end in enumerate(groups):
        # Train/test split percentage
        perc = np.round((end / total) * 100, decimals=1)
        sf.loc[i, 'Train% / Test%'] = f'{perc:.1f} / {100-perc:.1f}'

        # Retrieve what this split date is
        if end != total:
            sf.loc[i, 'Split Date'] = df.index[end]

        # TRAIN
        train  = df.iloc[0:end]
        counts = train[target].value_counts()
        neg    = counts.get(0, 0)
        pos    = counts.get(1, 0)

        sf.loc[i, ('Train', 'Total'  , 0)] = neg
        sf.loc[i, ('Train', 'Total'  , 1)] = pos
        sf.loc[i, ('Train', 'Percent', 0)] = np.round((neg / labels[0])*100, decimals=1)
        sf.loc[i, ('Train', 'Percent', 1)] = np.round((pos / labels[1])*100, decimals=1)

        # TEST
        test   = df.iloc[end:]
        counts = {} if test.empty else test[target].value_counts()
        neg    = counts.get(0, 0)
        pos    = counts.get(1, 0)

        sf.loc[i, ('Test' , 'Total'  , 0)] = neg
        sf.loc[i, ('Test' , 'Total'  , 1)] = pos
        sf.loc[i, ('Test' , 'Percent', 0)] = np.round((neg / labels[0])*100, decimals=1)
        sf.loc[i, ('Test' , 'Percent', 1)] = np.round((pos / labels[1])*100, decimals=1)

    return sf

def _split_single_regression(df, n=10, **kwargs):
    """
    Parameters
    ----------
    df: pandas.core.DataFrame
        DataFrame to split
    n: int
        Number of splits to perform where step is 100/n

    Returns
    -------
    """
    # [S]plit [F]rame, DataFrame to store possible splits
    sf = pd.DataFrame(columns=['Train', 'Test'], index=range(n-1))
    sf['Train% / Test%'] = np.nan
    sf['Split Date']     = np.nan

    total  = df.shape[0]
    groups = [int(i*total/n) for i in range(1, n)]

    for i, end in enumerate(groups):
        perc = np.round((end / total) * 100, decimals=1)
        sf.loc[i, 'Train% / Test%'] = f'{perc:.1f} / {100-perc:.1f}'

        if end != total:
            sf.loc[i, 'Split Date'] = df.index[end]

        sf.loc[i, 'Train'] = df.iloc[0:end].shape[0]
        sf.loc[i, 'Test' ] = df.iloc[end:].shape[0]

    return sf

def interact(df):
    """
    Parameters
    ----------

    Returns
    -------
    """
    def _split():
        """
        """
        Logger.info('Divide the data into N equal sized pieces')

        while True:
            n = input('n = ')
            try:
                n = int(n)

                return func(
                    df     = df,
                    n      = n,
                    target = config.target
                )
            except:
                Logger.error(f'The input value is not an integer: {value!r}')
                return _split()

    def _select():
        """
        """
        Logger.info(f'\n{sf}')
        Logger.info('Please select an index to use as the split date from the above table.')
        Logger.info('If you would like to choose a different N, enter -1.')

        while True:
            index = input('index: ')
            try:
                index = int(index)

                if index == -1:
                    return interact(df)
                elif index >= sf.shape[0]:
                    Logger.error(f'Index out of bounds: {index}')
                else:
                    return str(sf['Split Date'].loc[index])
            except:
                Logger.error(f'The input value is not an integer: {value!r}')
                return _select

    config = Config()

    func = _split_single_regression
    if config.classification:
        func = _split_single_classification

    sf = _split()

    return _select()

def main():
    """

    Returns
    -------
    bool
        Returns True if the function completed successfully
    """
    # Retrieve the config object
    config = Config()

    if config.input.multi:
        for key in tqdm(config.input.multi, position=1):
            df = pd.read_hdf(config.input.file, key)
            df = process(df, features)
            df.to_hdf(config.output.file, f'windows/{key}')
    else:
        df = pd.read_hdf(config.input.file, 'windows')

        if config.interactive:
            config.split_date = interact(df)

        train, test = split(df, config.split_date)
        train, test = select(df, train, test, n_jobs=config.get('cores', 1))

    Logger.info(f'Saving to {config.output.file} under key select/[train,test]')
    train.to_hdf(config.output.file, 'select/train')
    test .to_hdf(config.output.file, 'select/test')

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/config.yaml',
                                            help     = 'Path to a config.yaml file'
    )
    parser.add_argument('-s', '--section',  type     = str,
                                            default  = 'subselect',
                                            help     = 'Section of the config to use'
    )

    args = parser.parse_args()

    try:
        utils.init(args)

        state = main()

        Logger.info('Finished successfully')
    except Exception as e:
        Logger.exception('Failed to complete')
