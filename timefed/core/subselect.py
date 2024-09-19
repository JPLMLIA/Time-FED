import argparse
import logging
import numpy  as np
import pandas as pd
import xarray as xr

from typing import Tuple

from pathlib import Path

from mlky    import Config
from tqdm    import tqdm
from tsfresh import select_features

from timefed.core.extract import verify
from timefed.utils        import utils

Logger = logging.getLogger('timefed/core/subselect.py')


def xr_split(ds, date):
    """
    Simple date splitting on xarray Datasets

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to operate on
    date : str
        Datetime string to split with

    Returns
    -------
    train, test : Tuple[xr.Dataset, xr.Dataset]
        Train and test subsets
    """
    date = pd.to_datetime(date)

    train = ds.datetime.isel(index=0) <= date
    test = ds.datetime.isel(index=0) > date
    train, test = ds.sel(windowID=train.data), ds.sel(windowID=test.data)

    return train, test


def select(train: pd.DataFrame, test: pd.DataFrame, target: str='label', n_jobs: int=1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Uses tsfresh.select_features to select relevant features from the tsfresh feature
    extraction. See for more:
    https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_selection.html#tsfresh.feature_selection.selection.select_features

    Parameters
    ----------
    train : pd.DataFrame
        DataFrame containing the training data.
    test : pd.DataFrame
        DataFrame containing the test data.
    target : str, optional, default='label'
        Name of the target column.
    n_jobs : int, optional, default=1
        Number of parallel jobs to run.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the modified train and test DataFrames.

    Notes
    -----
    The function selects features based on the training set and ensures that the same features
    are kept in the test set as well.

    Configuration Options:
    - `chunksize`: Chunk size used for feature selection.
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


def _split_multi_classification(data: dict, n: int = 1, **kwargs) -> pd.DataFrame:
    """
    Creates a train/test split table for interactive use for multi track classification cases.

    Parameters
    ----------
    data : dict
        Dictionary containing information about the streams.
    n : int, optional, default=1
        Number of splits to create.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the splits.

    Notes
    -----
    This function splits the multi-classification dataset based on the unique edges of the streams.
    It computes various statistics for each split regarding the distribution of the classes.
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
    sf['Train% / Test%'] = ''
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
    train  = sf[('Train', 'Total')].sum(axis=1) / totals
    test   = sf[('Test', 'Total')].sum(axis=1) / totals
    sf['Train% / Test%'] = [f'{t0:.2%} / {t1:.2%}' for t0, t1 in zip(train, test)]

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


def _split_single_classification(data: pd.DataFrame, n: int = 10, target: str = 'label', **kwargs) -> pd.DataFrame:
    """
    Creates a train/test split table for interactive use for single track classification cases.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame to be split.
    n : int, optional, default=10
        Number of splits to create.
    target : str, optional, default='label'
        Name of the target column.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the splits.

    Notes
    -----
    This function splits the data into 'n' consecutive segments and computes various statistics
    for each split regarding the distribution of the target variable.

    Additional keyword arguments can be passed but are not used in this function.

    Example
    -------
    >>> data = {
    ...     'stream1': {'start': '2024-01-01', 'end': '2024-01-10', 'neg': 20, 'pos': 30},
    ...     'stream2': {'start': '2024-01-02', 'end': '2024-01-09', 'neg': 15, 'pos': 25},
    ...     'stream3': {'start': '2024-01-03', 'end': '2024-01-08', 'neg': 25, 'pos': 35},
    ...     'stream4': {'start': '2024-01-04', 'end': '2024-01-07', 'neg': 30, 'pos': 40},
    ...     'stream5': {'start': '2024-01-05', 'end': '2024-01-06', 'neg': 10, 'pos': 20},
    ... }
    >>> _split_multi_classification(data, n=2)
    [TODO]
    """
    # [S]plit [F]rame, DataFrame to store possible splits
    sf = pd.DataFrame(columns=pd.MultiIndex.from_product([['Train', 'Test'], ['Total', 'Percent'], [0, 1]]), index=range(n-1))
    sf['Train% / Test%'] = ''
    sf['Split Date']     = ''

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


def _split_single_regression(data: pd.DataFrame, n: int = 10, **kwargs) -> pd.DataFrame:
    """
    Creates a train/test split table for interactive use for single track regression cases.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame to be split.
    n : int, optional, default=10
        Number of splits to create.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the splits.

    Notes
    -----
    This function splits the data into 'n' consecutive segments for regression tasks.
    It calculates the size of the train and test sets and the percentage each represents
    relative to the total dataset size.

    Example
    -------
    >>> data = pd.DataFrame(np.random.randn(100, 2), columns=['feature1', 'feature2'])
    >>> _split_single_regression(data, n=5)
    [TODO]
    """
    # [S]plit [F]rame, DataFrame to store possible splits
    sf = pd.DataFrame(columns=['Train', 'Test'], index=range(n-1))
    sf['Train% / Test%'] = ''
    sf['Split Date']     = ''

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
    Interacts with the user to split the data into train/test sets and select a split date.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame to be split.

    Returns
    -------
    str
        The selected split date as a string.

    Notes
    -----
    This function interacts with the user to perform data splitting and selection of a split date.
    """
    def _split(func):
        """
        Wrapper function for splitting data into chunks depending on one of four use cases:
        - Single track regression
        - Single track classification
        - Multi track regression [Not Implemented]
        - Multi track classification

        Parameters
        ----------
        func : function
            The function to split the data per the use case.

        Returns
        -------
        pd.DataFrame
            The split frame returned by the split function.
        """
        Logger.info('Divide the data into N chunks')
        Logger.info('If this is a single stream, these are equal-sized split steps')
        Logger.info('If this is a multi stream, these are every Nth split choice')

        while True:
            n = input('[User Input] Please select an N: ')
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
        Helper function to select a split date.

        Returns
        -------
        str
            The selected split date as a string.
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


    if Config.subselect.multi:
        func = _split_multi_regression
        if Config.model.type == 'classification':
            func = _split_multi_classification
    else:
        func = _split_single_regression
        if Config.model.type == 'classification':
            func = _split_single_classification

    sf = _split(func)

    return _select()


def cube(dfs, target=None, classification=False):
    """
    """
    hold = []
    for df in dfs:
        if classification:
            label = [int(df[target].any())]
            df = df.drop(columns=[target])

        # Retrieve what will become the single index of this window
        idx = df.index[-1]

        ds = df.reset_index().to_xarray()
        ds = ds.expand_dims(window=[idx])

        if classification:
            ds[target] = ('window', label)

        hold.append(ds)

    return xr.merge(hold)


def main():
    """
    TODO

    Returns
    -------
    bool
        Returns True if the function completed successfully
    """
    if Config.subselect.multi:
        if not Path(metadata := Config.subselect.metadata).exists():
            Logger.error('The multisteam case requires a metadata file produced by extract.py')
            return

        Logger.info(f'Reading file {metadata}')
        data = utils.load_pkl(metadata)

    elif Config.extract.method == 'passthrough':
        Logger.info(f'Reading file {Config.extract.output}/preprocess/complete.nc')
        ds = xr.load_dataset(f'{Config.extract.output}/preprocess/complete.nc')

        train, test = xr_split(ds, Config.subselect.split_date)

        Logger.info(f'Writing to file {Config.extract.output}/preprocess/complete.nc under select/[train,test]')
        train.to_netcdf(Config.extract.output, mode='a', format='NETCDF4', group='select/train')
        test .to_netcdf(Config.extract.output, mode='a', format='NETCDF4', group='select/test')

        return True
    else:
        Logger.info(f'Reading file {Config.subselect.file}[extract/complete]')
        data = pd.read_hdf(Config.subselect.file, 'extract/complete')

        if Config.model.type == 'classification':
            Logger.debug(f'This is a classification dataset using the target: {Config.model.target}')
            if Config.model.target not in data:
                Logger.error(f'The target does not exist in the dataset: {Config.model.target}')
                return

            keys = data[Config.model.target].value_counts().sort_index().keys()
            if len(keys) > 2:
                Logger.error('Classification only supports binary values')
                return
            if not all(keys == [0, 1]):
                Logger.error('Classification only supports binary values')
                return

    date = Config.subselect.split_date
    if Config.subselect.interactive:
        date = interact(data)
    date = pd.Timestamp(date)

    Logger.info(f'Using split date: {date}')

    # Now load the splits
    if Config.subselect.multi:
        model_targ = Config.model.target
        model_type = Config.model.type == 'classification'

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
            track = pd.read_hdf(Config.subselect.file, key)
            index = track.index.name
            streams.append(track)

        Logger.info(f'Retrieved {len(streams)} tracks as the train set')
        Logger.debug(f'Tracks: {train}')

        if Config.subselect.output == 'pandas':
            Logger.info('Merging to a single dataframe')
            index   = streams[0].index.name
            streams = [stream.reset_index() for stream in streams]
            train   = pd.concat(streams, axis=0, ignore_index=True).set_index(index)

        elif Config.subselect.output == 'xarray':
            Logger.info('Casting train dataframes to datasets')
            train = cube(streams,
                target = model_targ,
                classification = model_type
            )

        streams = []
        for key in test:
            track = pd.read_hdf(Config.subselect.file, key)
            streams.append(track)

        Logger.info(f'Retrieved {len(streams)} tracks as the test set')
        Logger.debug(f'Tracks: {test}')

        if Config.subselect.output == 'pandas':
            Logger.info('Merging to a single dataframe')
            index   = streams[0].index.name
            streams = [stream.reset_index() for stream in streams]
            test    = pd.concat(streams, axis=0, ignore_index=True).set_index(index)

            Logger.info('Verifying the train dataset does not have NaNs')
            train = verify(train)
            Logger.info('Verifying the test dataset does not have NaNs')
            test  = verify(test)

        elif Config.subselect.output == 'xarray':
            Logger.info('Casting dataframes to datasets')
            test = cube(streams,
                target = model_targ,
                classification = model_type
            )

    else:
        Logger.debug(f'Split date selected: {date}')

        # Subselect features to process on
        if (features := Config.subselect.features):
            if isinstance(features, list):
                copy = data[features]
            elif isinstance(features, str):
                copy = data.filter(regex=features)
            else:
                Logger.error(f'Features provided cannot be interpreted as either a list or a regex string: {features}')

            if copy.empty:
                Logger.error('Feature subselection returned empty')
                return

            # Add target back in
            copy[Config.model.target] = data[Config.model.target]

            # Track the dropped columns so they can be saved elsewhere
            dropped = set(data) - set(copy)

            # Override and set as the data to train/test and select on
            data = copy

        train = data.query('index <  @date')
        test  = data.query('index >= @date')

        # TODO: Implement correctly
        # if Config.subselect.save_dropped:
        #     train.to_hdf(Config.subselect.file, 'select/train')
        #     test .to_hdf(Config.subselect.file, 'select/test')

        Logger.debug(f'Train shape: {train.shape} ({train.shape[0]/data.shape[0]*100:.2f}%)')
        Logger.debug(f'Test  shape: {test.shape} ({test.shape[0]/data.shape[0]*100:.2f}%)')


    if Config.subselect.tsfresh:
        if Config.subselect.output == 'pandas':
            train, test = select(train, test,
                target = Config.model.target,
                n_jobs = Config.subselect.get('cores', 1))
        else:
            Logger.warning(f'tsfresh feature selection cannot be executed on {Config.subselect.output} objects, skipping')

    if Config.subselect.output == 'pandas':
        Logger.info(f'Saving to {Config.subselect.file} under key select/[train,test]')
        train.to_hdf(Config.subselect.file, key='select/train')
        test .to_hdf(Config.subselect.file, key='select/test')

    else:
        Config.subselect.file = Config.subselect.file.replace('.h5', '.nc')
        Logger.info(f'Saving to {Config.subselect.file} under keys select/[train,test]')
        train.to_netcdf(Config.subselect.file, mode='a', format='NETCDF4', group='select/train')
        test .to_netcdf(Config.subselect.file, mode='a', format='NETCDF4', group='select/test')

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
