import argparse
import logging
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics  import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)

# Import utils first to set the logger
import utils
import plots

logger = logging.getLogger('mloc/classify.py')

def train_and_test(model, train, test, label, features):
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
    if len(features) > 20:
        logger.debug(f'Using features {features[:20]} + {len(features)-20} more')
    else:
        logger.debug(f'Using features {features}')
    logger.debug(f'Using label {label}')

    model.fit(train[features], train[label])

    pred     = model.predict(test[features])
    r2       = r2_score(test[label], pred)
    rms      = mean_squared_error(test[label], pred, squared=False)
    perc_err = mean_absolute_percentage_error(test[label], pred)

    logger.debug(f'r2 score      = {r2}')
    logger.debug(f'RMS error     = {rms}')
    logger.debug(f'percent error = {perc_err}')

    return pred, {
        'R2'  : r2,
        'RMSE': rms,
        'MAPE': perc_err
    }

def build_model(config, shift=None):
    """
    """
    if shift:
        train = pd.read_hdf(config.input.file, f'{config.input.key}/historical_{shift}_min/train')
        test  = pd.read_hdf(config.input.file, f'{config.input.key}/historical_{shift}_min/train')
    else:
        train = pd.read_hdf(config.input.file, f'{config.input.key}/train')
        test  = pd.read_hdf(config.input.file, f'{config.input.key}/test')

    logger.info('Creating, training, and testing the model')

    model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)

    # Retrieve features list, exclude the label from it
    features = config.features.select
    if features is None:
        features = list(train.columns)

    # Remove the label from the features list
    if config.label in features:
        features = list(set(features) - set([config.label]))

    # Exclude certain features
    if config.features.exclude:
        features = list(set(features) - set(config.features.exclude))

    # Drop rows that contain a NaN in any column
    if config.dropna:
        train = train[~train.isnull().any(axis=1)]
        test  =  test[ ~test.isnull().any(axis=1)]

    # Train and test the model
    pred, scores = train_and_test(model, train, test, config.label, features)

    # Save predictions and model
    if config.output:
        key = config.output.keys.forecasts
        if shift:
            key += f'/H{shift}'

        preds = pd.Series(index=test.index, data=pred)
        preds.to_hdf(config.output.file, key)

        # Save model via pickle
        output = f'{config.output.models}/{config.label}'
        if shift:
            output += f'_H{shift}_min'

        output += '.pkl'
        utils.save_pkl(output, model)

    return pred, scores

@utils.timeit
def classify(config):
    """
    Builds a model, trains and tests against it, then creates plots for the run.

    Parameters
    ----------
    config : utils.Config
        Config object defining arguments for classify
    """
    if config.input.historical:
        # Disable plot generation when multiple models are being built
        config.plots.enabled = False

        # Create the scores dataframe
        df = pd.DataFrame(index=config.input.historical)
        df.index.name = 'Forecast'

        for shift in config.input.historical:
            logger.debug(f'Building model with historical shift {shift}')
            pred, scores = build_model(config, shift=shift)

            # Save scores in dataframe
            for score, value in scores.items():
                if score not in df:
                    df[score] = np.nan
                df[score][shift] = value
    else:
        pred, scores = build_model(config)
        df = pd.DataFrame(scores, index=['Score'])

    # Save scores
    if config.output:
        df.to_hdf(config.output.file, config.output.keys.scores)

    # If plotting is enabled, run the functions and save output if given
    if config.plots.enabled:
        plots.scatter_with_errors(test[config.label], pred, lambda a, b: a-b,     name='true_diff', config=config)
        plots.scatter_with_errors(test[config.label], pred, lambda a, b: (a-b)/a, name='perc_diff', config=config)

        plots.errors_in_time(test[config.label], pred, config=config)
        plots.importances(model, features, config=config)

        plots.histogram_errors(test[config.label], pred, lambda a, b: (a-b)/a, config=config)

        plots.local_synchrony(train, config=config)

        plots.date_range(test[config.label], pred, config=config)


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

        classify(config)

        logger.info('Finished successfully')
    except Exception as e:
        logger.exception('Failed to complete')
