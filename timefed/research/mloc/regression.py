import argparse
import logging
import numpy as np
import os
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics  import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)

# Import utils first to set the logger
from mloc import utils
from mloc.research import plots

logger = logging.getLogger('mloc/research/classify.py')

def train_and_test(model, train, test, label, fit=True):
    """
    Trains and tests a model

    Parameters
    ----------
    train : pandas.DataFrame
        The dataframe containing training data for the model
    test : pandas.DataFrame
        The dataframe containing testing data for the model
    label : str
        The name of the column that is used as the label
    features : list of str
        List of columns to use as features. If not provided, defaults to all
        columns in the train dataframe

    Returns
    -------
    pred : list
        List of predicted values
    r2 : float
        The r2 score of the model
    """
    # if len(features) > 20:
    #     logger.debug(f'Using features {features[:20]} + {len(features)-20} more')
    # else:
    #     logger.debug(f'Using features {features}')
    logger.debug(f'Using label {label}')

    if fit:
        model.fit(train.drop(columns=[label]), train[label])

    pred = model.predict(test.drop(columns=[label]))

    stats = {}
    try:
        r2       = r2_score(test[label], pred)
        rms      = mean_squared_error(test[label], pred, squared=False)
        perc_err = mean_absolute_percentage_error(test[label], pred)

        logger.debug(f'r2 score      = {r2}')
        logger.debug(f'RMS error     = {rms}')
        logger.debug(f'percent error = {perc_err}')

        stats = {
            'R2'  : r2,
            'RMSE': rms,
            'MAPE': perc_err
        }
    except:
        pass

    return pred, stats

def build_model(config, shift=None):
    """
    """
    if shift is not None:
        train = pd.read_hdf(config.input.file, f'{config.input.key}/historical_{shift}_min/train')
        test  = pd.read_hdf(config.input.file, f'{config.input.key}/historical_{shift}_min/test')
    else:
        train = pd.read_hdf(config.input.file, f'{config.input.key}/train')
        test  = pd.read_hdf(config.input.file, f'{config.input.key}/test')

    logger.info('Creating, training, and testing the model')

    # Load model via pickle
    model = None
    fit   = False
    if config.output:
        output = f'{config.output.models}/{config.label}'
        if shift is not None:
            output += f'_H{shift}_min'
        output += '.pkl'

        if os.path.exists(output):
            if config.remake is True:
                pass
            else:
                if shift is not None:
                    if shift not in config.remake:
                        model = utils.load_pkl(output)
                elif not config.remake:
                    model = utils.load_pkl(output)

    if not model:
        logger.debug('Creating new model')
        model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
        fit   = True

    # # Retrieve features list, exclude the label from it
    # features = config.features.select
    # if features is None:
    #     features = list(train.columns)
    #
    # # Remove the label from the features list
    # if config.label in features:
    #     features = list(set(features) - set([config.label]))
    #
    # # Exclude certain features
    # if config.features.exclude:
    #     features = list(set(features) - set(config.features.exclude))

    # Drop rows that contain a NaN in any column
    orig = train.index.size
    train = train[~train.isnull().any(axis=1)] # Train ALWAYS drops

    logger.debug(f'Dropping NaNs reduced the data by {(1-train.index.size/orig)*100:.2f}%')
    if train.isna().any().any():
        logger.debug(f'Percent of NaNs in columns that had NaNs:\n{(shift[train.columns[train.isna().any()]].isna().sum() / train.index.size) * 100}')

    if config.dropna:
        orig = test.index.size
        test = test[~test.isnull().any(axis=1)]

        logger.debug(f'Dropping NaNs reduced the data by {(1-test.index.size/orig)*100:.2f}%')
        if test.isna().any().any():
            logger.debug(f'Percent of NaNs in columns that had NaNs:\n{(shift[test.columns[test.isna().any()]].isna().sum() / test.index.size) * 100}')

    # Drop rows that have a value for the label
    if config.inverse_drop:
        test = test.loc[test[config.label].isnull()]

    # Train and test the model
    pred, scores = train_and_test(model, train, test, config.label, fit)

    # Cast predicted values to Series
    pred = pd.Series(index=test.index, data=pred)

    # Save predictions and model
    if config.output:
        key = config.output.keys.forecasts
        if shift is not None:
            key += f'/H{shift}'

        pred.to_hdf(config.output.file, key)

        # Save model via pickle
        utils.save_pkl(output, model)

    return train, test, pred, scores, model

@utils.timeit
def regress(config):
    """
    Builds a model, trains and tests against it, then creates plots for the run.

    Parameters
    ----------
    config : utils.Config
        Config object defining arguments for classify
    """
    if config.input.historical:
        # Create the scores dataframe
        df = pd.DataFrame(index=config.input.historical)
        df.index.name = 'Forecast'

        for shift in config.input.historical:
            logger.debug(f'Building model with historical shift {shift}')
            train, test, pred, scores, model = build_model(config, shift=shift)

            # Save scores in dataframe
            for score, value in scores.items():
                if score not in df:
                    df[score] = np.nan
                df[score][shift] = value

            # If plotting is enabled, run the functions and save output if given
            if config.plots.enabled:
                if not os.path.exists(config.plots.directory):
                    os.mkdir(config.plots.directory)
                if not os.path.exists(f'{config.plots.directory}/H{shift}'):
                    os.mkdir(f'{config.plots.directory}/H{shift}')
                bak = config.plots.directory
                config.plots.directory += f'/H{shift}'
                plots.generate_plots(test, pred, model, config, train)
                config.plots.directory = bak
    else:
        train, test, pred, scores, model = build_model(config)
        df = pd.DataFrame(scores, index=['Score'])

        # If plotting is enabled, run the functions and save output if given
        if config.plots.enabled:
            plots.generate_plots(test, pred, model, config, train)

    # Save scores
    if config.output:
        df.to_hdf(config.output.file, config.output.keys.scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/config.yaml',
                                            help     = 'Path to a config.yaml file'
    )
    parser.add_argument('-s', '--section',  type     = str,
                                            default  = 'classify',
                                            help     = 'Section of the config to use'
    )

    args = parser.parse_args()

    try:
        config = utils.Config(args.config, args.section)

        regress(config)

        logger.info('Finished successfully')
    except Exception as e:
        logger.exception('Failed to complete')
