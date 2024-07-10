import argparse
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path

from mlky import (
    Config,
    Sect
)
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

from timefed.utils import utils
from timefed.utils.plots import importances


Logger = logging.getLogger('timefed/model.py')


def regress_score(model, data, name):
    """
    """
    config   = Config()
    truth    = data[Config.model.target]
    data     = data.drop(columns=Config.model.target)
    preds    = truth.copy()
    preds[:] = model.predict_proba(data)

    scores = Sect({
        'r2'      : r2_score(truth, preds),
        'rms'     : mean_squared_error(truth, preds),
        'perc_err': mean_absolute_percentage_error(truth, preds)
    })

    Logger.info(f'Scores for {name}:')
    utils.align_print(scores, print=Logger.info, prepend='  ')

    importances(model, data.columns, print_only=True)

    # Return as dict instead of Section
    return scores._data, preds


def class_score(model, data, name, multiclass_scores=False):
    """
    """
    config   = Config()
    truth    = data[Config.model.target]
    data     = data.drop(columns=Config.model.target)
    preds    = truth.copy()
    preds[:] = model.predict_proba(data)

    scores = Sect({
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
        TN, FP, FN, TP = scores.confusion_matrix.ravel()
        scores.TN, scores.FP, scores.FN, scores.TP = TN, FP, FN, TP

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
                index   = pd.MultiIndex.from_product([['truth']    , list(range(value.shape[0]))]),
                columns = pd.MultiIndex.from_product([['predicted'], list(range(value.shape[1]))])
            )
            Logger.info(f'{key} = \n{cm}')
        else:
            Logger.info(f'{key:9} = {value}')

    scores.roc_curve = roc_curve(truth, preds)

    Logger.info(f'Classification Report:\n{classification_report(truth, preds, target_names=sorted(pd.unique(preds).astype(str)))}')

    # plot_roc(truth, preds)

    importances(model, data.columns, print_only=True)

    # Return as dict instead of Section
    return scores._data, preds


@utils.timeit
def main():
    """
    Builds a model, trains and tests against it, then creates plots for the run.

    Parameters
    ----------
    config : utils.Config
        Config object defining arguments for classify
    """
    # Retrieve this script's configuration section to reduce verbosity of code
    C = Config.model

    data = Sect({
        'train': pd.read_hdf(C.file, 'select/train'),
        'test' : pd.read_hdf(C.file, 'select/test')
    })

    for kind, df in data.items():
        if df.isna().any(axis=None):
            Logger.info(f'The {kind} set was detected to have NaNs. Please correct this and rerun. See debug for more info.')
            Logger.debug('The following columns contained a NaN:')
            utils.align_print(dict(enumerate(df.columns[df.isna().any()])), print=Logger.debug, prepend='- ', delimiter=':')
            return 1

    # Either load an existing model file
    if Path(path := C.model.file).exists() and not C.model.overwrite:
        Logger.info(f'Loading existing model: {path}')
        model = utils.load_pkl(path)

    # or create a new one
    else:
        C.model.fit = True

        if C.type == 'classification':
            model = RandomForestClassifier(**C.model.params)
            score = class_score
        else:
            model = RandomForestRegressor(**C.model.params)
            score = regress_score

    # Only train the model if set
    if C.model.fit:
        model.fit(
            data.train.drop(columns=[C.target]),
            data.train[C.target]
        )

        # Save the newly trained model
        if C.output.model:
            exists = Path(path := C.model.file).exists()
            if not exists or (exists and C.model.overwrite):
                utils.save_pkl(path, model)

    # Score the test dataset
    (scores := {})['test'], predicts = score(model, data.test, 'test')

    if C.output.predicts:
        predicts.to_hdf(C.output.predicts, 'predicts/test')

    if C.train_scores:
        scores['train'], predicts = score(model, data.train, 'train')

        if C.output.predicts:
            predicts.to_hdf(C.output.predicts, 'predicts/train')

    if C.output.scores:
        utils.save_pkl(C.output.scores, scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/Config.model.yaml',
                                            help     = 'Path to a Config.model.yaml file'
    )
    parser.add_argument('-s', '--section',  type     = str,
                                            default  = 'model',
                                            help     = 'Section of the config to use'
    )

    args = parser.parse_args()

    try:
        utils.init(args)

        code = main()

        Logger.info('Finished successfully')
    except Exception as e:
        Logger.exception('Failed to complete')
