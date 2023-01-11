import argparse
import logging
import numpy as np
import os
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.metrics import (
    # Classification metrics
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    # Regression metrics
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)

# Import utils first to set the logger
from mloc import utils
from mloc.research import plots

logger = logging.getLogger('timefed/model.py')

def regress_score(model, data, name):
    """
    """
    config = Config()

    truth = data[config.label]
    data  = data.drop(columns=config.label)
    preds = model.predict(data)

    scores = Section('scores', {
        'r2'      : r2_score(truth, preds),
        'rms'     : mean_squared_error(truth, preds),
        'perc_err': mean_absolute_percentage_error(truth, preds)
    })

    Logger.info(f'Scores for {name}:')
    align_print(scores, print=Logger.info, prepend='  ')

    importances(model, data.columns, print_only=False)

    # Return as dict instead of Section
    return scores._data

def class_score(model, data, name, multiclass_scores=False):
    """
    """
    config = Config()

    truth = data[config.label]
    data  = data.drop(columns=config.label)
    preds = model.predict(data)

    scores = Section('scores', {
        'accuracy'        : accuracy_score(truth, preds),
        'precision'       : precision_score(truth, preds),
        'recall'          : recall_score(truth, preds),
        'roc_auc'         : roc_auc_score(truth, preds),
        'confusion_matrix': confusion_matrix(truth, preds),
    })

    if multiclass_scores:
        scores.TP = TP = scores.confusion_matrix.diagonal()
        scores.FP = FP = scores.confusion_matrix.sum(axis=0) - TP
        scores.FN = FN = scores.confusion_matrix.sum(axis=1) - TP
        scores.TN = TN = scores.confusion_matrix.sum() - (FP + FN + TP)
    else:
        scores.TP = TP = scores.confusion_matrix[1, 1]
        scores.FP = FP = scores.confusion_matrix[1, 0]
        scores.FN = FN = scores.confusion_matrix[0, 1]
        scores.TN = TN = scores.confusion_matrix[0, 0]

    scores.TPR = TP/(TP+FN) # Sensitivity, hit rate, recall, or true positive rate
    scores.TNR = TN/(TN+FP) # Specificity or true negative rate
    scores.PPV = TP/(TP+FP) # Precision or positive predictive value
    scores.NPV = TN/(TN+FN) # Negative predictive value
    scores.FPR = FP/(FP+TN) # Fall out or false positive rate
    scores.FNR = FN/(TP+FN) # False negative rate
    scores.FDR = FP/(TP+FP) # False discovery rate
    scores.ACC = (TP+TN)/(TP+FP+FN+TN) # Overall accuracy

    Logger.info(f'Scores for {name}:')
    for key, value in scores.items():
        # Special format for CM
        if key == 'confusion_matrix':
            cm = pd.DataFrame(value,
                index   = pd.MultiIndex.from_product([['predicted'], list(range(value.shape[0]))]),
                columns = pd.MultiIndex.from_product([['truth']    , list(range(value.shape[1]))])
            )
            Logger.info(f'{key} = \n{cm}')
        else:
            Logger.info(f'{key:9} = {value}')

    scores.roc_curve = roc_curve(truth, preds)

    Logger.info(f'Classification Report:\n{classification_report(truth, preds, target_names=sorted(pd.unique(preds).astype(str)))}')

    # plot_roc(truth, preds)

    importances(model, data.columns, print_only=False)

    # Return as dict instead of Section
    return scores._data

def classify():
    """
    """
    config = Config()

    # Load in the train/test dataframes
    train = pd.read_hdf(config.input.file, f'{config.input.key}/train')
    test  = pd.read_hdf(config.input.file, f'{config.input.key}/test')

    # Log some analysis on the input data
    analyze(train, test, config.label)

    # Create the model and fit
    model = RandomForestClassifier(**config.RandomForestClassifier)
    model.fit(train.drop(columns=config.label), train[config.label])

    scores = {
        'test' : score(model,  test, 'test'),
        'train': score(model, train, 'train')
    }

    return scores

@utils.timeit
def main():
    """
    Builds a model, trains and tests against it, then creates plots for the run.

    Parameters
    ----------
    config : utils.Config
        Config object defining arguments for classify
    """
    # Retrieve the config object
    config = Config()

    data = Section('data', {
        'train': pd.read_hdf(config.input.file, 'select/train'),
        'test' : pd.read_hdf(config.input.file, 'select/test')
    })

    for kind, df in data.items():
        if df.isna().any(axis=None):
            Logger.info('The {kind}ing set was detected to have NaNs. Please correct this and rerun. See debug for more info.')
            Logger.debug('The following columns contained a NaN:')
            align_print(dict(enumerate(df.columns[df.isna().any()])), print=Logger.debug, prepend='- ', delimiter=':')
            return 1

    if config.classification:
        Logger.info('This is a classification run, using RandomForestClassifier')
        if config.model.load:
            Logger.info(f'Loading model from config.output.model: {config.output.model}')
            utils.load_pkl(config.output.model)
        else:
            Logger.info('Creating new model')
            model = RandomForestClassifier(**config.model.params)
            config.model.fit = True

        if config.model.fit:
            model.fit(data.train.drop(columns=[config.target]), data.train[config.target])

        scores = {
            'test' : class_score(model,  data.test, 'test'),
            'train': class_score(model, data.train, 'train')
        }
    else:
        Logger.info('This is a regression run, using RandomForestRegressor')
        if config.model.load:
            Logger.info(f'Loading model from config.output.model: {config.output.model}')
            utils.load_pkl(config.output.model)
        else:
            Logger.info('Creating new model')
            model = RandomForestRegressor(**config.model.params)
            config.model.fit = True

        if config.model.fit:
            model.fit(data.train.drop(columns=[config.target]), data.train[config.target])

        scores = {
            'test' : regress_score(model,  data.test, 'test'),
            'train': regress_score(model, data.train, 'train')
        }



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/config.yaml',
                                            help     = 'Path to a config.yaml file'
    )
    parser.add_argument('-s', '--section',  type     = str,
                                            default  = 'extract',
                                            help     = 'Section of the config to use'
    )

    args = parser.parse_args()

    try:
        utils.init(args)

        with logging_redirect_tqdm():
            code = main()

        Logger.info('Finished successfully')
    except Exception as e:
        Logger.exception('Failed to complete')
