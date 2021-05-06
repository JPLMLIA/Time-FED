"""
Plotting functions for classify.py
"""
import argparse
import logging
import matplotlib.pyplot as plt
import numpy   as np
import pandas  as pd
import seaborn as sns

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats             import gaussian_kde
from sklearn.metrics         import mean_squared_error

import utils

logger = logging.getLogger('mloc/plots.py')

sns.set_context('talk')
sns.set_style('darkgrid')


def protect(func):
    """
    Protects the calling function from any exceptions that may be raised by func

    Parameters
    ----------
    func : function
        Function to wrap in a try/except
    """
    def wrap(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except:
            logger.exception(f'Failed to generate plot: {func.__name__}')

    return wrap

@protect
def local_synchrony(df, config):
    """

    """
    # Get the plot configs for this plot type
    pconf = config.plots.local_synchrony

    # Retrieve the desired features
    feat1 = df[pconf.feature1]
    feat2 = df[pconf.feature2]

    corr = feat1.rolling(window=pconf.window_size, center=True).corr(feat2)

    w, h = pconf.figsize
    fig, axes = plt.subplots(3, 1, figsize=(w*1, h*3), sharex=True)

    ax = axes[0]
    ax.plot(feat1)
    ax.set_ylabel(feat1.name)

    ax = axes[1]
    ax.plot(feat2)
    ax.set_ylabel(feat2.name)

    ax = axes[2]
    corr.plot(ax=ax)
    ax.set_ylabel('Pearson r')

    plt.suptitle('Data and rolling window correlation')
    plt.tight_layout()
    if config.plots.directory:
        plt.savefig(f'{config.plots.directory}/{pconf.file}')
    else:
        plt.show()

@protect
def histogram_errors(true, pred, error_func, config):
    """
    Bins the true values of the label column and plots with the error per bin

    Parameters
    ----------
    true : pandas.Series
    pred : list
    error_func : func
    config : utils.Config
    """
    pconf = config.plots.histogram_errors

    # Calculate errors using the function given
    # min   = pconf.min or np.min(true)
    # max   = pconf.max or np.max(true)
    # bins  = pconf.bins or max - min + 1
    # bins  = np.linspace(min, max, bins)

    df    = pd.DataFrame({'floor': np.floor(true), 'errs': error_func(true, pred)})
    group = df.groupby('floor')
    means = group.mean()
    stds  = group.std()
    x = sorted(df.floor.unique())

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=pconf.figsize)
    ax.bar(x, means['errs'], yerr=stds['errs'])
    ax.set_xticks(np.arange(0, np.max(x), 4))

    # Set labels
    ax.set_xlabel(f'{config.label} floor ({config.plots.units[config.label]})')
    ax.set_ylabel('% error')
    ax.set_title(f'Errors per {config.label} binned')

    plt.tight_layout()
    if config.plots.directory:
        plt.savefig(f'{config.plots.directory}/{pconf.file}')
    else:
        plt.show()

@protect
def errors_in_time(true, pred, config):
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
    # Get the plot configs for this plot type
    pconf = config.plots.errors_in_time

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
    ax.set_ylabel(f'Error {config.label} ({config.plots.units[config.label]})')

    ax = axes[2]
    ax.plot(true.index, (true-pred)/true, 'bx')
    ax.set_xlabel('Datetime')
    ax.set_ylabel(f'% error {config.label} ({config.plots.units[config.label]})')

    plt.tight_layout()
    if config.plots.directory:
        plt.savefig(f'{config.plots.directory}/errors_in_time.png')
    else:
        plt.show()

@protect
def scatter_with_errors(true, pred, error_func, name, config):
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
    name : str
        Name of the error function to be used in the filename of the plot
    """
    def scatter(x, y, ax, label, minx, maxx, miny, maxy, zeroes=True):
        _x = np.linspace(x.min(), x.max(), pconf.samples)

        # plot
        ax.scatter(x, y, edgecolor='k', c='seagreen', s=pconf.size, alpha=pconf.alpha)

        # Draw base line
        if zeroes:
            ax.plot(_x, np.zeros(_x.shape), 'r-')
        else:
            ax.plot(x, x, 'r-')

        # Set limits and labels
        ax.set_xlim([minx, maxx])
        ax.set_ylim([miny, maxy])
        ax.set_xlabel(f'Actual {config.label} ({config.plots.units[config.label]})')
        ax.set_ylabel(f'{label} {config.label} ({config.plots.units[config.label]})')

    def contour(m1, m2, ax, minx, maxx, miny, maxy, zeroes=True):
        x, y = np.mgrid[
            minx : maxx : pconf.step_length,
            miny : maxy : pconf.step_length,
        ]
        pos  = np.vstack([x.ravel(), y.ravel()])
        vals = np.vstack([m1, m2])
        kern = gaussian_kde(vals)

        z = np.reshape(kern(pos).T, x.shape)

        # Draw base line
        if zeroes:
            ax.plot(x, np.zeros(x.shape), 'r-')
        else:
            ax.plot(x, x, 'r-')

        image   = ax.contourf(x, y, z, 20)#, vmin=pconf.colorscale.min, vmax=pconf.colorscale.max)
        divider = make_axes_locatable(ax)
        colorax = divider.append_axes('right', size='5%', pad=0.05)

        ax.set_xlim([minx, maxx])
        ax.set_ylim([miny, maxy])

        fig.colorbar(image, cax=colorax)

    # Get the plot configs for this plot type
    pconf = config.plots.scatter_with_errors

    # Create subplots
    w, h = pconf.figsize
    fig, axes = plt.subplots(2, 2, figsize=(w*2, h*2))

    # Get the axis limits
    minx, maxx = pconf.x.min, pconf.x.max
    miny, maxy = pconf.y.min, pconf.y.max

    try:
        # Plot scatter and contour for actual vs predicted
        scatter(true, pred, axes[0, 0], 'Predicted', zeroes=False, minx=minx, maxx=maxx, miny=miny, maxy=maxy)
        contour(true, pred, axes[0, 1],              zeroes=False, minx=minx, maxx=maxx, miny=miny, maxy=maxy)
    except:
        logger.exception('Failed to generate scatter_with_errors actual plots')

    # Plot scatter and contour for actual vs errors
    errs = error_func(true, pred)
    val  = np.isfinite(errs)
    errs = errs[val]
    miny = errs.min() - 1
    maxy = errs.max() + 1

    logger.debug(f'scatter_with_errors | Using miny={miny}, maxy={maxy}')

    try:
        scatter(true[val], errs, axes[1, 0], 'Error', zeroes=True, minx=minx, maxx=maxx, miny=miny, maxy=maxy)
        contour(true[val], errs, axes[1, 1],          zeroes=True, minx=minx, maxx=maxx, miny=miny, maxy=maxy)
    except:
        logger.exception('Failed to generate scatter_with_errors error plots')

    plt.tight_layout()
    if config.plots.directory:
        plt.savefig(f'{config.plots.directory}/{name}.{pconf.file}')
    else:
        plt.show()

@protect
def importances(model, features, config):
    """
    Plots the important features of the given model

    Parameters
    ----------
    model : any
        A fitted model
    features : list
        List of possible features
    """
    # Get the plot configs for this plot type
    pconf = config.plots.importances

    # Retriev the important features
    imports = model.feature_importances_
    stddev  = np.std([est.feature_importances_ for est in model.estimators_], axis=0)
    indices = np.argsort(imports)[::-1]

    logger.info('Feature ranking:')
    for i in range(len(features)):
        logger.info(f'- {i+1}. {features[indices[i]]:20} ({imports[indices[i]]})')

    xaxis   = range(min([pconf.number or len(features), len(features)]))
    indices = indices[:len(xaxis)]

    # Plot
    fig, ax = plt.subplots(figsize=pconf.figsize)
    ax.bar(x=xaxis, height=imports[indices], yerr=stddev[indices], align='center', color='r')

    # Set ylim
    ax.set_ylim([0, 1])
    ax.set_yticks([0, .25, .5, .75, 1])

    # Set xlim and ticks
    ax.set_xlim([-1, len(xaxis)])
    ax.set_xticks(xaxis)
    ax.tick_params(axis='x', labelrotation=pconf.labelrotation)

    # Set labels
    ax.set_xticklabels([features[i] for i in indices])
    ax.set_title(pconf.title)

    plt.tight_layout()
    if config.plots.directory:
        plt.savefig(f'{config.plots.directory}/{pconf.file}')
    else:
        plt.show()

@protect
def date_range(true, pred, config):
    """
    """
    # Get the plot configs for this plot type
    pconf = config.plots.date_range

    # Generate a plot for each date range given
    for date, dconf in pconf.dates:
        # Subselect
        start, end = dconf.start, dconf.end
        logger.debug(f'Plotting between {start} to {end}')

        true_sub = true.loc[(start <= true.index) & (true.index < end)]
        pred_sub = pred.loc[(start <= pred.index) & (pred.index < end)]
        print(true_sub, pred_sub)

        # Plot
        fig, ax = plt.subplots(figsize=pconf.figsize)
        ax.plot(true_sub, 'g.', label='true')
        ax.plot(pred_sub, 'r.', label='predicted')

        title = f'True vs Predicted between {start} to {end}'
        if 'rms' in pconf.metrics:
            rms = mean_squared_error(true_sub.values, pred_sub.values, squared=False)
            title += f'\nRMS Error = {rms:.4f}'
            ax.plot(np.abs(true_sub-pred_sub), 'b.', label='Absolute error')

        ax.legend()
        ax.set_ylabel(f'{config.label} ({config.plots.units[config.label]})')
        ax.set_xlabel(f'Month-Day Hour ({config.plots.units.datetime})')
        ax.set_title(title)

        # Save
        plt.tight_layout()
        if config.plots.directory:
            plt.savefig(f'{config.plots.directory}/{date}.{pconf.file}')
        else:
            plt.show()

def generate_plots(test, pred, model, config):
    """
    Generates all plots
    """
    scatter_with_errors(test[config.label], pred.values, lambda a, b: a-b,     name='true_diff', config=config)
    scatter_with_errors(test[config.label], pred.values, lambda a, b: (a-b)/a, name='perc_diff', config=config)

    errors_in_time(test[config.label], pred.values, config=config)
    importances(model, test.columns, config=config)

    histogram_errors(test[config.label], pred.values, lambda a, b: (a-b)/a, config=config)
    # local_synchrony(train, config=config)

    date_range(test[config.label], pred, config=config)


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
    parser.add_argument('-ki', '--inkey',   type     = str,
                                            required = True,
                                            help     = 'The key for the test dataframe in the file of the input section of the config'
    )
    parser.add_argument('-ko', '--outkey',  type     = str,
                                            required = True,
                                            help     = 'The key for the forecasted values in the file of the output section of the config'
    )
    parser.add_argument('-m', '--model',    type     = str,
                                            metavar  = '/path/to/model.pkl',
                                            help     = 'The model to use for feature importance plotting'
    )

    args = parser.parse_args()

    try:
        config = utils.Config(args.config, args.section)

        test = pd.read_hdf(config.input.file, args.inkey).dropna()
        pred = pd.read_hdf(config.output.file, args.outkey)

        model = None
        if args.model:
            model = utils.load_pkl(args.model)

        generate_plots(test, pred, model, config)

        logger.info('Finished successfully')
    except Exception as e:
        logger.exception('Failed to complete')
