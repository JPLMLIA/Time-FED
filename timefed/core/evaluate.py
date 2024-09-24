#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:32:15 2024

@author: marchett
"""

import os, logging
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
from timefed.utils import evaluate_tools as ev
import warnings
Logger = logging.getLogger('timefed/model.py')
warnings.filterwarnings('always')
import seaborn as sns
plt.style.use('ggplot')
sns.set_style('whitegrid')

def evaluate_deep(base_dir, version, key = 'predicts/test'):
    
    #version = 'lstm_NEU_v3'
    #keys = ['windows', 'events']
    
    plots_dir = os.path.dirname(base_dir)
    Logger.info(f'Saving plots in {plots_dir}.')
    #read output results
    windows_df = pd.read_hdf(f'{base_dir}', key)
    preds = windows_df['pred']
    true_labels = windows_df['actual']
    proba = windows_df['proba']
    
    ############# plotting probability histograms
    # probability histogram
    pos_class = (true_labels == 1)
    neg_class = (true_labels == 0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.05)  # adjust space between Axes

    # plot the same data on both Axes
    ax1.hist(proba[pos_class], bins = 20, histtype = 'barstacked', density = True,
             label = ' true positives', alpha = 0.5);
    ax1.hist(proba[neg_class], bins = 20, histtype = 'barstacked', density = True,
             label = ' true negatives', alpha = 0.5);
    c1,_,_ =ax2.hist(proba[pos_class], bins = 20, histtype = 'barstacked', density = True,
             label = ' true positives', alpha = 0.5);
    c2,_,_ =ax2.hist(proba[neg_class], bins = 20, histtype = 'barstacked', density = True,
             label = ' true negatives', alpha = 0.5);
    ax1.legend()
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    # zoom-in / limit the view to different portions of the data
    b1 = np.max([c1[1:-1], c2[1:-1]])
    ax1.set_ylim(b1*0.3, np.max([c1, c2]) * 1.1)  # outliers only
    ax2.set_ylim(0, 0.5)  # most of the data
    
    # plt.figure()
    # plt.hist(proba[pos_class], bins = 20, histtype = 'barstacked', density = True,
    #          label = ' true positives', alpha = 0.5);
    # plt.hist(proba[neg_class], bins = 20, histtype = 'barstacked', density = True,
    #          label = ' true negatives', alpha = 0.5);
    # plt.ylim((0, 5), top = 1)
    plt.xlabel('probability'); plt.ylabel('density')
    ax1.set_title(f'classifier probability distributions\n{version}, windows')
    plt.savefig(f'{plots_dir}/windows_prob_hist_{version}.png')
    plt.close()
    
    ############# plotting f-1 score
    thresholds = np.linspace(0, 1)
    tpr_ = []; fnr_ = []
    f1 = []
    for z in thresholds:
        preds_cutoff = (proba > z).astype(int)
        f1.append(metrics.f1_score(true_labels, preds_cutoff))
        tp, fn, fp, tpr, fnr, fpr = ev.scores(true_labels, preds_cutoff)
        tpr_.append(tpr); fnr_.append(fnr)
    plt.figure()
    plt.plot(thresholds, tpr_, label = 'TPR')
    plt.plot(thresholds, fnr_, label = 'FNR')
    plt.plot(thresholds, f1, label = 'f1-score')
    plt.scatter(thresholds[np.argmax(f1)], f1[np.argmax(f1)], color = 'r')
    plt.axvline(thresholds[np.argmax(f1)], ls = '--', color = 'r')
    plt.xlabel('thresholds'); plt.ylabel('rate');
    plt.legend()
    plt.title(f'metrics across probability cutoff thresholds\n{version}, windows');
    plt.savefig(f'{plots_dir}/windows_thresholds_{version}.png')
    plt.close()
    
    
    ############# plotting ROC curves
    mask_nan = np.isnan(proba)
    #plot evaluation 
    ev.curve_plots(true_labels[~mask_nan], preds[~mask_nan], 
                   proba[~mask_nan], version, 0.5, np.max(f1), 'windows',
                   new = True)
    plt.savefig(f'{plots_dir}/windows_roc_pr_curves.png')
    plt.close()
    
    ev.confusion_matrix(true_labels, preds, version, 'windows')
    plt.savefig(f'{plots_dir}/windows_conf_matrix_{version}.png')
    plt.close()

        

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str)
    #parser.add_argument('--compare_dir', type=str)

    args = parser.parse_args()
    evaluate_deep(**vars(args))
