# Copyright 2025, by the California Institute of Technology. ALL RIGHTS RESERVED.
# United States Government Sponsorship acknowledged. Any commercial use must be
# negotiated with the Office of Technology Transfer at the California Institute of
# Technology.
#
# This software may be subject to U.S. export control laws. By accepting this software,
# the user agrees to comply with all applicable U.S. export laws and regulations. User
# has the responsibility to obtain export licenses, or other export authority as may be
# required before exporting such information to foreign countries or providing access
# to foreign persons.

import os
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics


Logger = logging.getLogger('timefed/model.py')
warnings.filterwarnings('always')

plt.style.use('ggplot')
sns.set_style('whitegrid')


def plotProbLogged(pos, neg, version=None, save=None):
    """
    Creates a logged bar stacked histogram plot of the probability of the positive and
    negative classes predicted by the LSTM

    Parameters
    ----------
    pos : pandas.DataFrame
        Positive class with "probability" as a column
    neg : pandas.DataFrame
        Negative class with "probability" as a column
    version : str, default=None
        Version to append to the title
    save : pathlib.Path, default=None
        Directory to save the figure to as a png file

    Returns
    -------
    fig, ax
        Returns the fig and ax for further modifications
    """
    fig, ax = plt.subplots()

    ax.hist(neg.probability, bins=20, histtype='barstacked', density=True, label='True Negative', alpha=0.5)
    ax.hist(pos.probability, bins=20, histtype='barstacked', density=True, label='True Positive', alpha=0.5)

    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel('probability')
    ax.set_ylabel('density')

    title = 'LSTM Probability Distribution of Windows'
    if version:
        title += f'\nVersion: {version}'
    ax.set_title(title)

    plt.tight_layout()
    if save:
        plt.savefig(save / "logged_hist.png")

    return fig, ax


def plotProbZoomed(pos, neg, version=None, save=None):
    """
    Creates bar stacked histogram plots of the probability of the positive and negative
    classes predicted by an LSTM. Splits the histogram to zoom in on the majority

    Parameters
    ----------
    pos : pandas.DataFrame
        Positive class with "probability" as a column
    neg : pandas.DataFrame
        Negative class with "probability" as a column
    version : str, default=None
        Version to append to the title
    save : pathlib.Path, default=None
        Directory to save the figure to as a png file

    Returns
    -------
    fig, tuple
        Returns the figure and axes for further modifications
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=1)

    ax1.hist(neg.probability, bins=20, histtype='barstacked', density=True, label='True Negative', alpha=0.5)
    ax1.hist(pos.probability, bins=20, histtype='barstacked', density=True, label='True Positive', alpha=0.5)

    ax1.legend()

    c1, *_ = ax2.hist(neg.probability, bins=20, histtype='barstacked', density=True, label='True Negative', alpha=0.5)
    c2, *_ = ax2.hist(pos.probability, bins=20, histtype='barstacked', density=True, label='True Positive', alpha=0.5)

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    # zoom-in / limit the view to different portions of the data
    b1 = np.max([c1[1:-1], c2[1:-1]])
    ax1.set_ylim(b1*0.3, np.max([c1, c2]) * 1.1)  # outliers only
    ax2.set_ylim(0, 0.5)  # most of the data

    ax2.set_xlabel('probability')
    ax2.set_ylabel('density')

    title = 'LSTM Probability Distribution of Windows'
    if version:
        title += f'\nVersion: {version}'
    ax1.set_title(title)

    plt.tight_layout()
    if save:
        plt.savefig(save / "zoomed_hist.png")

    return fig, (ax1, ax2)


def plotConfusionMatrix(df, version=None, save=None):
    """
    Plots a confusion matrix

    Parameters
    ----------
    df : pandas.DataFrame
        Contains "truth" and "predict" columns
    version : str, default=None
        Version to append to the title
    save : pathlib.Path, default=None
        Directory to save the figure to as a png file

    Returns
    -------
    img : sklearn.metrics.ConfusionMatrixDisplay
        Returns the ConfusionMatrixDisplay for further modifications
    """
    lbls = np.array([["True Negative", "False Negative"], ["False Positive", "True Positive"]])
    cm   = metrics.confusion_matrix(df.truth, df.predict)
    acc  = metrics.accuracy_score(df.truth, df.predict)
    prec = metrics.precision_score(df.truth, df.predict)
    rec  = metrics.recall_score(df.truth, df.predict)
    f1   = metrics.f1_score(df.truth, df.predict)

    title = f'Accuracy={acc:.2f}, Precision={prec:.2f}, Recall={rec:.2f}, F1={f1:.2f}'
    if version:
        title += f'\nVersion: {version}'

    percent = cm / cm.sum(axis=0)

    img = metrics.ConfusionMatrixDisplay(cm)
    img.plot()
    img.ax_.set_title(title, fontsize=12)
    img.ax_.grid(False)
    for obj in img.text_.ravel():
        pos  = obj.get_position()
        perc = percent[pos]
        text = obj.get_text()
        obj.set_text(f"{lbls[pos]}\n{text} ({perc:.2%})")

    plt.tight_layout()
    if save:
        plt.savefig(save / "confusion_matrix.png")

    return img


def cm_scores(truth, predict):
    """
    Calculates the following scores:
    - True Positive
    - True Negative
    - False Positive
    - False Negative
    - True Positive Rate
    - True Negative Rate
    - False Negative Rate
    - False Positive Rate

    Parameters
    ----------
    truth : pandas.Series
        True labels
    predict : pandas.Series
        Predicted labels

    Returns
    -------
    dict
        Contains the scores
    """
    cm = metrics.confusion_matrix(truth, predict)

    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]

    tpr = tp / (tp + fn) # Recall / Sensitivity
    tnr = tn / (tn + fp) # Specificity
    fpr = fp / (fp + tn) #
    fnr = fn / (tp + fn) #

    return {
        "true pos": tp,
        "true neg": tn,
        "false pos": fp,
        "false neg": fn,
        "true pos rate": tpr,
        "true neg rate": tnr,
        "false pos rate": fpr,
        "false neg rate": fnr
    }


def thresholdScores(df, thresholds=np.linspace(0, 1)):
    """
    Calculates the TPR, FNR, F1 scores given various thresholds

    Parameters
    ----------
    df : pandas.DataFrame
        Contains "truth", "predict", and "probability" columns
    thresholds : list, default=np.linspace(0, 1)
        Thresholds to calculate score for

    Returns
    -------
    TPR, FNR, F1, thresholds : tuple[list, list, list, iterable]
        Returns the scores for each threshold in the iterable
    """
    tpr, fnr, f1 = [], [], []
    for z in thresholds:
        cutoff = df.query("probability > @z")

        if cutoff.empty:
            rates = (1.0, 0.0)
        else:
            scores = cm_scores(cutoff.truth, cutoff.predict)
            rates  = scores['true pos rate'], scores['false neg rate']

        tpr.append(rates[0])
        fnr.append(rates[1])
        f1.append(metrics.f1_score(cutoff.truth, cutoff.predict))

    return tpr, fnr, f1, thresholds


def plotProbThresholds(df, version=None, save=None):
    """
    Plots probability metrics across certain cutoff thresholds

    Parameters
    ----------
    df : pandas.DataFrame
        Contains "truth", "predict", and "probability" columns
    version : str, default=None
        Version to append to the title
    save : pathlib.Path, default=None
        Directory to save the figure to as a png file

    Returns
    -------
    fig, ax
        Returns the fig and ax for further modifications
    """
    tpr, fnr, f1, thresholds = thresholdScores(df)

    fig, ax = plt.subplots()
    ax.plot(thresholds, tpr, label="True Positive Rate")
    ax.plot(thresholds, fnr, label="False Negative Rate")
    ax.plot(thresholds, f1,  label="F-1 Score")

    peak = np.argmax(f1)
    ax.scatter(thresholds[peak], f1[peak], color='r')
    ax.axvline(thresholds[peak], ls='--', color='r', linewidth=1)

    ax.set_xlabel('thresholds')
    ax.set_ylabel('rate')
    ax.legend()

    title = 'Metrics Across Probability Cutoff Thresholds'
    if version:
        title += f'\nVersion: {version}'
    ax.set_title(title)

    plt.tight_layout()
    if save:
        plt.savefig(save / "thresholds.png")

    return fig, ax


def plotCurves(df, a=0.5, f1=None, version=None, save=None):
    """
    Plots Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves

    Parameters
    ----------
    df : pandas.DataFrame
        Contains "truth", "predict", and "probability" columns
    a : float, default=0.5
        TODO
    f1 : float, default=None
        F1 score. Will calculate one if not provided
    version : str, default=None
        Version to append to the title
    save : pathlib.Path, default=None
        Directory to save the figure to as a png file

    Returns
    -------
    fig, axes : tuple[matplotlib.figure.Figure, tuple[matplotlib.axes._axes.Axes, matplotlib.axes._axes.Axes]]
        Returns the fig and axes for further modifications
    """
    fpr, tpr, thresholds = metrics.roc_curve(df.truth, df.probability)
    roc_auc = metrics.auc(fpr, tpr)
    prec, rec, alpha_pr = metrics.precision_recall_curve(df.truth, df.probability)

    index_pr = np.argmin(np.abs(alpha_pr - a))
    avgPrec  = metrics.average_precision_score(df.truth, df.probability)

    if f1 is None:
        f1 = np.max(thresholdScores(df)[2])

    # If thresholds is in descending order, invert it for the search
    invert = 1
    if thresholds[0] > thresholds[-1]:
        invert = -1

    # Find the cutoff index
    cutoff = np.searchsorted(invert*thresholds, invert*a) - 1
    fpr_co = fpr[cutoff]
    tpr_co = tpr[cutoff]

    cutoff = np.searchsorted(invert*thresholds, invert*f1) - 1
    fpr_f1 = fpr[cutoff]
    tpr_f1 = tpr[cutoff]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, color='darkorange', lw=2, zorder=0, label=f'ROC, AUC={roc_auc:0.2f}')
    axes[0].scatter(fpr_co, tpr_co, color='r', s=40, zorder=1)
    axes[0].scatter(fpr_f1, tpr_f1, color='g', s=40, zorder=1, label=f'Cutoff F1 {f1:.2f} (FPR={fpr_f1:.2f}, TPR={tpr_f1:.2f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend(loc='lower right')

    title = 'Receiver Operating Characteristic (ROC) Curve'
    if version:
        title += f'\nVersion: {version}'
    axes[0].set_title(title)

    axes[1].plot(rec, prec, color='b', lw=2, zorder=0, label=f'AP={avgPrec:0.2f}')
    axes[1].scatter(rec[index_pr], prec[index_pr], color='r', s=40, zorder=40)
    axes[1].set_xlabel('Recall TP/(TP+FN)')    # How many of positives were correct
    axes[1].set_ylabel('Precision TP/(TP+FP)') # How many are actually positive
    axes[1].set_title(f'Precision-Recall (PR) curve')
    axes[1].legend(loc='lower left')

    plt.tight_layout()
    if save:
        plt.savefig(save / "roc_pr_curves.png")

    return fig, axes


def generatePlots(file, version=None, output=None):
    """
    Evaluates the results of an LSTM model

    Parameters
    ----------
    file : str
        Path to an H5 file from a TimeFED LSTM output
    version : str, default=None
        Version to append to plot titles
    output : str
        Path to a directory save generated plots to
    """
    df = pd.read_hdf(file, 'predicts/test')

    classes = df.groupby(df['truth'])
    neg = df.loc[classes.groups[0]]
    pos = df.loc[classes.groups[1]]

    if output:
        output = Path(output)
        output.mkdir(exist_ok=True, parents=True)

    plotProbLogged(pos, neg, version=version, save=output)
    plotProbZoomed(pos, neg, version=version, save=output)
    plotProbThresholds(df, version=version, save=output)
    plotCurves(df, version=version, save=output)
    plotConfusionMatrix(df, version=version, save=output)
