"""
Summary: plotting scripts for notebooks

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
sns.set_context('talk')


#
# SUBROUTINES
#

def error_perc(truth, pred):
    return np.divide(truth-pred,truth)

def error_diff(truth, pred):
    return truth-pred

def density_estimation(m1, m2, xmin, xmax, ymin, ymax):
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    return X, Y, Z

def scatter_with_errors(truth, preds, error_func, xmin, xmax, ymin, ymax):
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    s = 25
    a = 0.4
    ax[0, 0].scatter(truth, preds, edgecolor='k', c="cornflowerblue", s=s, alpha=a)
    x = np.linspace(truth.min(), truth.max(), 1000)
    ax[0, 0].plot(x, x, 'r-')
    ax[0, 0].set_xlabel("Actual r0")
    ax[0, 0].set_ylabel("Predicted r0")
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])

    X, Y, Z = density_estimation(truth, preds, xmin, xmax, ymin, ymax)
    ax[0, 1].plot(x, x, 'r-')
    im01 = ax[0, 1].contourf(X, Y, Z, 20)
    divider01 = make_axes_locatable(ax[0, 1])
    cax01 = divider01.append_axes('right', size='5%', pad=0.05)
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    fig.colorbar(im01, cax=cax01)

    # now plot with errors
    #
    errs = error_func(truth, preds)
    ymin = np.min(errs) - 1
    ymax = np.max(errs) + 1

    ax[1, 0].scatter(truth, errs, edgecolor='k', c="forestgreen", s=s, alpha=a)
    ax[1, 0].plot(x, np.zeros(x.shape), 'r-')
    ax[1, 0].set_xlabel("Actual r0")
    ax[1, 0].set_ylabel("Error r0")
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])

    Xerr, Yerr, Zerr = density_estimation(truth, errs, xmin, xmax, ymin, ymax)
    ax[1, 1].plot(x, np.zeros(x.shape), 'r-')
    im11 = ax[1, 1].contourf(Xerr, Yerr, Zerr, 20)
    divider11 = make_axes_locatable(ax[1, 1])
    cax11 = divider11.append_axes('right', size='5%', pad=0.05)
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    fig.colorbar(im11, cax=cax11)

    plt.show()

    def plot_errors_in_time(truth, preds):
        fig, ax = plt.subplots(3, 1)
        ax[0].plot(truth.index, truth, 'gx', label='actual')
        ax[0].plot(truth.index, preds, 'ro', label='predicted')
        ax[0].set_xlabel("Datetime")
        ax[0].set_ylabel("r0")
        ax[0].legend()

        ax[1].plot(truth.index, error_diff(truth, preds), 'bx')
        ax[1].set_xlabel("Datetime")
        ax[1].set_ylabel("error r0")

        ax[2].plot(truth.index, error_perc(truth, preds), 'bx')
        ax[2].set_xlabel("Datetime")
        ax[2].set_ylabel("perc error r0")
        plt.show()

    def plot_importance(forest, X, featnames):
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        for f in range(X.shape[1]):
            print(f"{f + 1}. {featnames[indices[f]]:20} ({importances[indices[f]]})")

        # Plot the impurity-based feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(X.shape[1]), [featnames[i] for i in indices], rotation='vertical')
        plt.xlim([-1, X.shape[1]])
        plt.show()

    def plot_overall_synchrony(feat1, feat2, feat1name, feat2name, r):
        f, ax = plt.subplots(2, 1, figsize=(7, 3), sharex=True)
        ax[0].plot(feat1, label=feat1name)
        ax[1].plot(feat2, label=feat2name)
        ax[1].set(title=f"Overall Pearson r = {np.round(r, 2)}");
        return

    def plot_local_synchrony(feat1, feat2, feat1name, feat2name, r_window_size=120):
        # Compute rolling window synchrony
        rolling_r = feat1.rolling(window=r_window_size, center=True).corr(feat2)
        f, ax = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
        ax[0].plot(feat1, label=feat1name)
        ax[1].plot(feat2, label=feat2name)
        rolling_r.plot(ax=ax[2])
        ax[0].set(ylabel=feat1name)
        ax[1].set(ylabel=feat2name)
        ax[2].set(ylabel='Pearson r')
        plt.suptitle("data and rolling window correlation")

    def print_pearsonr(feat1, feat2):
        r, p = stats.pearsonr(feat1, feat2)
        print(f"Scipy computed Pearson r: {r} and p-value: {p}")
        return r, p

def error_by_r0_histograms(truth, errs, r0min, r0max):
    plt.figure()
    bins = np.linspace(r0min, r0max, 81)
    d = {'r0_floor': np.floor(truth), 'errs': errs}
    df = pd.DataFrame(data=d)
    errs_by_r0_mean = df.groupby('r0_floor').mean()
    errs_by_r0_std = df.groupby('r0_floor').std()
    x = sorted(df['r0_floor'].unique())
    print(np.arange(0, max(x), 5))
    plt.bar(x, errs_by_r0_mean['errs'], yerr=errs_by_r0_std['errs'])
    plt.xticks(np.arange(0, max(x), 4))

    plt.xlabel('r0 floor')
    plt.ylabel('perc error')
    plt.show()
    return
