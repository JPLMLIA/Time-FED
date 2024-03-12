import argparse
import logging
import numpy  as np
import pandas as pd

from pathlib import Path

from mlky    import Config
from tqdm    import tqdm
from tsfresh import select_features

from timefed.core.extract import verify
from timefed.utils        import utils

Logger = logging.getLogger('timefed/core/subselect.py')


def select(train, test, target='label', n_jobs=1):
    """
    Selects relevant features from a tsfresh dataframe of features for a target
    label

    Parameters
    ----------


    Returns
    -------

    """
    original = train.shape[1] - 1

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

    new = train.shape[1] - 1
    Logger.info(f'Number of features selected by tsfresh: {new}/{original} ({new/original:.2%})')

    return train, test

def _split_multi_classification(data, n=1, **kwargs):
    """
    TODO

    Parameters
    ----------


    Returns
    -------

    """
    # Find the unique edges of the streams
    splits = set()
    for key, stream in data.items():
        splits.update([stream['start'], stream['end']])

    # Make sure none of these edges interrupt any other stream
    pop = set()
    for date in splits:
        for key, stream in data.items():
            if (stream['start'] < date) and (date < stream['end']):
                pop.update([date])

    splits = sorted(splits - pop)
    splits = sorted(set(splits) - set([splits[0], splits[-1]]))

    # [S]plit [F]rame, DataFrame to store possible splits
    sf = pd.DataFrame(columns=pd.MultiIndex.from_product([['Train', 'Test'], ['Total', 'Percent'], [0, 1]]), index=range(len(splits)))
    sf[:]                = 0
    sf['Train% / Test%'] = np.nan
    sf['Split Date']     = splits

    # For each split date, find the samples of each class for each side of the split
    for i, date in enumerate(splits):
        for key, stream in data.items():
            if stream['end'] <= date:
                sf.loc[i, ('Train', 'Total', 0)] += stream['neg']
                sf.loc[i, ('Train', 'Total', 1)] += stream['pos']
            else:
                sf.loc[i, ('Test', 'Total', 0)] += stream['neg']
                sf.loc[i, ('Test', 'Total', 1)] += stream['pos']

    # Percentage of samples in train and test (samples=pos+neg)
    totals = sf[['Train', 'Test']].sum(axis=1)
    train  = np.round(sf[('Train', 'Total')].sum(axis=1) / totals * 100, decimals=1)
    test   = np.round(sf[('Test', 'Total')].sum(axis=1) / totals * 100, decimals=1)
    sf['Train% / Test%'] = [f'{t0} / {t1}' for t0, t1 in zip(train, test)]

    # Percentage of each class to the total for that class (% of neg total, % of pos total)
    train0 = sf[('Train', 'Total', 0)]
    train1 = sf[('Train', 'Total', 1)]
    test0  = sf[('Test', 'Total', 0)]
    test1  = sf[('Test', 'Total', 1)]
    tneg = train0 + test0
    tpos = train1 + test1

    sf[('Train', 'Percent', 0)] = np.round((train0 / tneg * 100).astype(float), decimals=1)
    sf[('Train', 'Percent', 1)] = np.round((train1 / tpos * 100).astype(float), decimals=1)
    sf[('Test' , 'Percent', 0)] = np.round((test0 / tneg * 100).astype(float), decimals=1)
    sf[('Test' , 'Percent', 1)] = np.round((test1 / tpos * 100).astype(float), decimals=1)

    # Subselect some split dates to present
    sf = sf.iloc[list(range(0, sf.shape[0], int(sf.shape[0] / n)))]

    return sf

def _split_single_classification(data, n=10, target='label', **kwargs):
    """
    Parameters
    ----------
    data: pandas.core.DataFrame
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

    total  = data.shape[0]
    labels = data[target].value_counts()
    groups = [int(i*total/n) for i in range(1, n)]
    for i, end in enumerate(groups):
        # Train/test split percentage
        perc = np.round((end / total) * 100, decimals=1)
        sf.loc[i, 'Train% / Test%'] = f'{perc:.1f} / {100-perc:.1f}'

        # Retrieve what this split date is
        if end != total:
            sf.loc[i, 'Split Date'] = data.index[end]

        # TRAIN
        train  = data.iloc[0:end]
        counts = train[target].value_counts()
        neg    = counts.get(0, 0)
        pos    = counts.get(1, 0)

        sf.loc[i, ('Train', 'Total'  , 0)] = neg
        sf.loc[i, ('Train', 'Total'  , 1)] = pos
        sf.loc[i, ('Train', 'Percent', 0)] = np.round((neg / labels[0])*100, decimals=1)
        sf.loc[i, ('Train', 'Percent', 1)] = np.round((pos / labels[1])*100, decimals=1)

        # TEST
        test   = data.iloc[end:]
        counts = {} if test.empty else test[target].value_counts()
        neg    = counts.get(0, 0)
        pos    = counts.get(1, 0)

        sf.loc[i, ('Test' , 'Total'  , 0)] = neg
        sf.loc[i, ('Test' , 'Total'  , 1)] = pos
        sf.loc[i, ('Test' , 'Percent', 0)] = np.round((neg / labels[0])*100, decimals=1)
        sf.loc[i, ('Test' , 'Percent', 1)] = np.round((pos / labels[1])*100, decimals=1)

    return sf

def _split_multi_regression(data, n=1, **kwargs):
    pass

def _split_single_regression(data, n=10, **kwargs):
    """
    Parameters
    ----------
    data: pandas.core.DataFrame
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

    total  = data.shape[0]
    groups = [int(i*total/n) for i in range(1, n)]

    for i, end in enumerate(groups):
        perc = np.round((end / total) * 100, decimals=1)
        sf.loc[i, 'Train% / Test%'] = f'{perc:.1f} / {100-perc:.1f}'

        if end != total:
            sf.loc[i, 'Split Date'] = data.index[end]

        sf.loc[i, 'Train'] = data.iloc[0:end].shape[0]
        sf.loc[i, 'Test' ] = data.iloc[end:].shape[0]

    return sf

def interact(data):
    """
    TODO

    Parameters
    ----------

    Returns
    -------

    """
    def _split(func):
        """
        """
        Logger.info('Divide the data into N chunks')
        Logger.info('If this is a single stream, these are equal-sized split steps')
        Logger.info('If this is a multi stream, these are every Nth split choice')

        while True:
            n = input('n = ')
            try:
                n = int(n)

                return func(
                    data   = data,
                    n      = n,
                    target = Config.model.target
                )
            except:
                Logger.exception(f'The input value is not an integer: {n!r}')
                return _split(func)

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
                    return interact(data)
                elif index not in sf.index:
                    Logger.error(f'Invalid index choice: {index}')
                else:
                    return str(sf['Split Date'].loc[index])
            except:
                Logger.error(f'The input value is not an integer: {value!r}')
                return _select

    if Config.subselect.input.multi:
        func = _split_multi_regression
        if Config.model.type == 'classification':
            func = _split_multi_classification
    else:
        func = _split_single_regression
        if Config.model.type == 'classification':
            func = _split_single_classification

    sf = _split(func)

    return _select()


def main():
    """
    TODO

    Returns
    -------
    bool
        Returns True if the function completed successfully
    """
    if Config.subselect.input.multi:
        if not Path(metadata := Config.subselect.metadata).exists():
            Logger.error('The multisteam case requires a metadata file produced by extract.py')
            return 4
        data = utils.load_pkl(metadata)
    else:
        Logger.info(f'Reading file {Config.subselect.file}[extract/complete]')
        data = pd.read_hdf(Config.subselect.file, 'extract/complete')

        if Config.model.type == 'classification':
            Logger.debug(f'This is a classification dataset using the target: {Config.model.target}')
            if Config.model.target not in data:
                Logger.error(f'The target does not exist in the dataset: {Config.model.target}')
                return 1

            keys = data[Config.model.target].value_counts().sort_index().keys()
            if len(keys) > 2:
                Logger.error('Classification only supports binary values')
                return 2
            if not all(keys == [0, 1]):
                Logger.error('Classification only supports binary values')
                return 3

    date = Config.subselect.split_date
    if Config.subselect.interactive:
        date = interact(data)
    date = pd.Timestamp(date)

    if Config.subselect.input.multi:
        Logger.info('Loading streams into memory')
        train = []
        test  = []
        for key, stream in data.items():
            if stream['end'] <= date:
                train.append(key)
            else:
                test.append(key)

        streams = []
        for key in train:
            streams.append(
                pd.read_hdf(Config.subselect.file, f'windows/{key}').reset_index()
            )
        Logger.info('Merging train datasets together')
        train = pd.concat(streams, axis=0, ignore_index=True).set_index('index')

        streams = []
        for key in test:
            streams.append(
                pd.read_hdf(Config.subselect.file, f'windows/{key}').reset_index()
            )
        Logger.info('Merging test datasets together')
        test = pd.concat(streams, axis=0, ignore_index=True).set_index('index')

        Logger.info('Verifying the train dataset does not have NaNs')
        train = verify(train)
        Logger.info('Verifying the test dataset does not have NaNs')
        test  = verify(test)
    else:
        Logger.debug(f'Split date selected: {date}')

        train = data.query('index <  @date')
        test  = data.query('index >= @date')

        Logger.debug(f'Train shape: {train.shape} ({train.shape[0]/data.shape[0]*100:.2f}%)')
        Logger.debug(f'Test  shape: {test.shape} ({test.shape[0]/data.shape[0]*100:.2f}%)')

    train, test = select(train, test,
        target = Config.model.target,
        n_jobs = Config.subselect.get('cores', 1))

    Logger.info(f'Saving to {Config.subselect.file} under key select/[train,test]')
    train.to_hdf(Config.subselect.file, 'select/train')
    test .to_hdf(Config.subselect.file, 'select/test')

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

        if state is True:
            Logger.info('Finished successfully')
        else:
            Logger.error(f'Failed to complete with exit code: {state}')
    except Exception as e:
        Logger.exception('Failed to complete')
