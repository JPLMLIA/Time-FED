"""
Plotting functions for classify.py
"""
import logging
import matplotlib.pyplot as plt
import numpy   as np
import pandas  as pd
import seaborn as sns

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats             import gaussian_kde

logger = logging.getLogger('mloc/plots.py')

sns.set_context('talk')

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
    min   = pconf.min or np.min(true)
    max   = pconf.max or np.max(true)
    bins  = pconf.bins or max - min + 1
    bins  = np.linspace(min, max, bins)

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
    ax.set_xlabel(f'{config.label} floor')
    ax.set_ylabel('% error')

    plt.tight_layout()
    if config.plots.directory:
        plt.savefig(f'{config.plots.directory}/{pconf.file}')
    else:
        plt.show()

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
    ax.set_ylabel(f'Error {config.label}')

    ax = axes[2]
    ax.plot(true.index, (true-pred)/true, 'bx')
    ax.set_xlabel('Datetime')
    ax.set_ylabel(f'% error {config.label}')

    plt.tight_layout()
    if config.plots.directory:
        plt.savefig(f'{config.plots.directory}/errors_in_time.png')
    else:
        plt.show()

def _scatter_with_errors(true, pred, error_func, name, config):
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
    # Get the plot configs for this plot
    pconf = config.plots.scatter_with_errors

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=pconf.figsize)

    ax = axes[0]
    x  = np.linspace(true.min(), true.max(), 1000)

    ax.scatter(true, pred, edgecolor='k', c='cornflowerblue', s=pconf.size, alpha=pconf.alpha)
    ax.plot(x, x, 'r-')

    ax.set_xlabel('Actual r0')
    ax.set_ylabel('Predicted r0')
    ax.set_xticks(np.arange(0, 25))
    ax.set_yticks(np.arange(0, 25))

    ax = axes[1]
    ax.scatter(true, error_func(true, pred), edgecolor='k', c='forestgreen', s=pconf.size, alpha=pconf.alpha)
    ax.plot(x, np.zeros(x.shape), 'r-')

    ax.set_xlabel('Actual r0')
    ax.set_ylabel('Error r0')
    ax.set_xticks(np.arange(0, 25))

    plt.tight_layout()
    if config.plots.directory:
        plt.savefig(f'{config.plots.directory}/{name}.{pconf.file}')
    else:
        plt.show()

def scatter_with_errors(true, pred, error_func, name, config):
    """

    """
    def scatter(x, y, ax, label, minx, maxx, miny, maxy, zeroes=True):
        _x = np.linspace(x.min(), x.max(), pconf.samples)

        # plot
        ax.scatter(x, y, edgecolor='k', c='cornflowerblue', s=pconf.size, alpha=pconf.alpha)

        # Draw base line
        if zeroes:
            ax.plot(_x, np.zeros(_x.shape), 'r-')
        else:
            ax.plot(x, x, 'r-')

        # Set limits and labels
        ax.set_xlim([minx, maxx])
        ax.set_ylim([miny, maxy])
        ax.set_xlabel(f'Actual {config.label}')
        ax.set_ylabel(f'{label} {config.label}')

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

        image   = ax.contourf(x, y, z, 20)
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

    xaxis   = range(pconf.number or len(features))
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

def date_range(true, pred, config):
    """
    """
    # Get the plot configs for this plot type
    pconf = config.plots.date_range

    # Cast to series for easy subselection
    pred = pd.Series(pred, index=true.index)

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
        ax.legend()
        ax.set_ylabel(config.label)
        ax.set_title(f'True vs Predicted between {start} to {end}')

        # Save
        plt.tight_layout()
        if config.plots.directory:
            plt.savefig(f'{config.plots.directory}/{date}.{pconf.file}')
        else:
            plt.show()
