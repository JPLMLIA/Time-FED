import argparse
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
import os

from sklearn.ensemble import RandomForestRegressor

# Import utils to set the logger
import utils

logger = logging.getLogger(os.path.basename(__file__))

def plot_errors_in_time(true, pred, output=None):
    """
    Plots the errors of the predicted label with the true label

    Parameters
    ----------
    true : pandas.Series
        The true values of label
    pred : list
        The predicted values of label
    output : str
        Path to directory to write plot to
    """
    fig, axes = plt.subplots(3, 1, figsize=(20*1, 10*3))

    ax = axes[0]
    ax.plot(true.index, true, 'gx', label='actual')
    ax.plot(true.index, pred, 'ro', label='predicted')
    ax.set_xlabel('Datetime')
    ax.set_ylabel('r0')
    ax.legend()

    ax = axes[1]
    ax.plot(true.index, true-pred, 'bx')
    ax.set_xlabel('Datetime')
    ax.set_ylabel('error r0')

    ax = axes[2]
    ax.plot(true.index, (true-pred)/true, 'bx')
    ax.set_xlabel('Datetime')
    ax.set_ylabel('% error r0')

    plt.tight_layout()
    if output:
        plt.savefig(f'{output}/errors_in_time.png')
    else:
        plt.show()

def scatter_with_errors(true, pred, error_func, s=25, alpha=.4, figsize=(20, 10), output=None, name=None):
    """
    Creates scatter plots of label predicted vs label actual and label error vs
    label actual

    Parameters
    ----------
    true : pandas.Series
        The true values of label
    pred : list
        The predicted values from the model
    error_func : func
        The function to use to calculate the error
    s : int
        Size of the scatter dots
    alpha : float
        Alpha value for scatter dots
    figsize : tuple of int
        Size of the plot figure
    output : str
        Path to directory to write plot to
    name : str
        Name of the error function to be used in the filename of the plot
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    ax = axes[0]
    x  = np.linspace(true.min(), true.max(), 1000)

    ax.scatter(true, pred, edgecolor='k', c='cornflowerblue', s=s, alpha=alpha)
    ax.plot(x, x, 'r-')

    ax.set_xlabel('Actual r0')
    ax.set_ylabel('Predicted r0')
    ax.set_xticks(np.arange(0, 25))
    ax.set_yticks(np.arange(0, 25))

    ax = axes[1]
    ax.scatter(true, error_func(true, pred), edgecolor='k', c='forestgreen', s=s, alpha=alpha)
    ax.plot(x, np.zeros(x.shape), 'r-')

    ax.set_xlabel('Actual r0')
    ax.set_ylabel('Error r0')
    ax.set_xticks(np.arange(0, 25))

    plt.tight_layout()
    if output:
        plt.savefig(f'{output}/scatter_errors.{name}.png')
    else:
        plt.show()

def plot_importance(model, features, output=None):
    """
    Plots the important features of the given model

    Parameters
    ----------
    model : any
        A fitted model
    features : list
        List of possible features
    output : str
        Path to directory to write plot to
    """
    imports = model.feature_importances_
    stddev  = np.std([est.feature_importances_ for est in model.estimators_], axis=0)
    indices = np.argsort(imports)[::-1]

    logger.info('Feature ranking:')
    for i in range(len(features)):
        logger.info(f'- {i+1}. {features[indices[i]]:20} ({imports[indices[i]]})')

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.bar(x=range(len(features)), height=imports[indices], yerr=stddev[indices], align='center', color='r')
    ax.set_xlim([-1, len(features)])
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([features[i] for i in indices])
    ax.tick_params(axis='x', labelrotation=30)
    ax.set_title('Feature importances')

    plt.tight_layout()
    if output:
        plt.savefig(f'{output}/feature_importances.png')
    else:
        plt.show()

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
    logger.debug(f'Using features {features}')
    logger.debug(f'Using label {label}')

    model.fit(train[features], train[label])
    r2   = model.score(test[features], test[label])
    pred = model.predict(test[features])

    logger.debug(f'r2 score = {r2}')

    return pred, r2

def classify(train, test, label, features=None, exclude=None, dropna=True, plot=False, save=None, **kwargs):
    """
    Builds a model, trains and tests against it, then creates plots for the run.

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
    exclude : list of str
        If provided, removes these columns from the features list
    dropna : bool
        Drops rows that contain a NaN in any column
    plot : bool
        Enables plotting
    save : str
        Writes plots to the given directory string
    """
    logger.info('Creating, training, and testing the model')

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    # Retrieve features list, exclude the label from it
    if features is None:
        features = train.columns

    # Remove the label from the features list
    if label in features:
        features = set(features) - set([label])

    # Exclude certain features
    if exclude:
        features = set(features) - set(exclude)

    # Cast back to list
    if not isinstance(features, list):
        features = list(features)

    # Drop rows that contain a NaN in any column
    if dropna:
        train = train[~train.isnull().any(axis=1)]
        test  = test[~test.isnull().any(axis=1)]

    # Train and test the model
    pred, r2 = train_and_test(model, train, test, label, features)

    # If plotting is enabled, run the functions and save output if given
    if plot:
        scatter_with_errors(test[label], pred, lambda a, b: a-b,     output=save, name='true_diff')
        scatter_with_errors(test[label], pred, lambda a, b: (a-b)/a, output=save, name='perc_diff')

        plot_errors_in_time(test[label], pred, output=save)
        plot_importance(model, features, output=save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--input',    type     = str,
                                            required = True,
                                            metavar  = '/path/to/preprocess.h5',
                                            help     = 'Path to preprocess.h5 which is produced by preprocess.py'
    )
    parser.add_argument('-f', '--features', type     = str,
                                            nargs    = '+',
                                            help     = 'Selects which features to use for training and testing'
    )
    parser.add_argument('-e', '--exclude',  type     = str,
                                            nargs    = '+',
                                            help     = 'Selects features to drop before modeling'
    )
    parser.add_argument('-l', '--label',    type     = str,
                                            required = True,
                                            metavar  = 'VAR',
                                            help     = 'Selects which variable to use as the label'
    )
    parser.add_argument('-p', '--plot',     action   = 'store_true',
                                            help     = 'Enables plot generation'
    )
    parser.add_argument('-s', '--save',     type     = str,
                                            metavar  = '/plot/directory/',
                                            help     = 'Path to directory to save plots to; leave blank to not save plots'
    )

    args = parser.parse_args()

    try:
        train = pd.read_hdf(args.input, 'train')
        test  = pd.read_hdf(args.input, 'test')

        classify(train, test, **vars(args))

        logger.info('Finished successfully')
    except Exception as e:
        logger.exception('Failed to complete')
