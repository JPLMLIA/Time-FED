import argparse
import logging
import matplotlib.pyplot as plt
import pandas  as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics  import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    RocCurveDisplay
)

from timefed        import utils
from timefed.config import (
    Config,
    Section
)
from timefed.research.plots import importances

sns.set_context('talk')
sns.set_style('darkgrid')

Logger = logging.getLogger('timefed/classify.py')

def analyze(train, test, label):
    """
    """
    Logger.info(f'Train range: {min(train.index).date()} to {max(train.index).date()}')
    Logger.info(f'Test  range: {min(test.index).date()} to {max(test.index).date()}')

    counts = Section(data={
        'train': train[label].value_counts(),
        'test' : test[label].value_counts()
    })
    Logger.info(f'Label counts for train:\n{counts.train}')
    Logger.info(f'Percent label counts for train:\n{counts.train / train[label].count() * 100}')
    Logger.info(f'Label counts for test:\n{counts.test}')
    Logger.info(f'Percent label counts for test:\n{counts.test / test[label].count() * 100}')

def score(model, data, name, multiclass_scores=False):
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

def plot_confusion_matrix(config, cm):
    """
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    cmp = ConfusionMatrixDisplay(confusion_matrix=cm)
    cmp.plot(ax=ax, colorbar=False)
    ax.grid(False)
    ax.set_title('Confusion Matrix')

    if config.plots.directory:
        plt.savefig(f'{config.plots.directory}/{pconf.file}')
    else:
        plt.show()

def plot_roc(truth, predicted):
    """
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    display = RocCurveDisplay.from_predictions(truth, predicted, estimator_name='RandomForestClassifier')
    display.plot(ax=ax)


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
        utils.init(args)

        code = classify()

        Logger.info('Finished successfully')
    except Exception as e:
        Logger.exception('Failed to complete')
