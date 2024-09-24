#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:30:36 2024

@author: marchett
"""

import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import to_datetime
sns.set_style('whitegrid')


def scores(true_labels, train_preds):
    
    #acc = metrics.accuracy_score(true_labels, train_preds)
    cm = metrics.confusion_matrix(true_labels, train_preds)
    tpr = cm[1,1] / (cm[1,1] + cm[1,0])
    fnr =  cm[1,0] / (cm[1,1] + cm[1,0])
    fpr = cm[0,1] / (cm[0,1] + cm[0,0])
    #pr = metrics.precision_score(true_labels, train_preds)
    #rec = metrics.recall_score(true_labels, train_preds)
    tp = cm[1,1]; fn = cm[1,0]; fp = cm[0,1]
    # fpr, tpr, alpha = roc_curve(true_labels, proba)
    # roc_auc = auc(fpr, tpr)
    
    return tp, fn, fp, tpr, fnr, fpr


# plots ROC and PR curves
# ev.curve_plots(true_labels, preds, proba, version = 'v0', key = 'test', new = True)
def curve_plots(true_labels, pred_labels, proba, version, a = 0.5, f1 = None, key = '', 
                colors = None, main_dir = None, new = False):
 
    proba = np.hstack(proba)
    true_labels = np.hstack(true_labels)
    fpr, tpr, thresholds = roc_curve(true_labels, proba)
    roc_auc = auc(fpr, tpr)
    prec, rec, alpha_pr = precision_recall_curve(true_labels, np.hstack(proba))
    index_pr = np.argmin(np.abs(alpha_pr - a))
    average_precision = average_precision_score(true_labels, np.hstack(proba))
    
    if f1 is None:
        thresholds = np.linspace(0, 1)
        tpr_ = []; fnr_ = []
        f1 = []
        for z in thresholds:
            preds_cutoff = (proba > z).astype(int)
            f1.append(metrics.f1_score(true_labels, preds_cutoff))
            tp, fn, fp, tpr, fnr, fpr = scores(true_labels, preds_cutoff)
            tpr_.append(tpr); fnr_.append(fnr)
        f1 = np.max(f1)   
    
    cutoff_index = np.where(thresholds >= a)[0][-1]
    fpr_cutoff = fpr[cutoff_index]
    tpr_cutoff = tpr[cutoff_index]
    
    cutoff_index = np.where(thresholds >= f1)[0][-1]
    fpr_f1 = fpr[cutoff_index]
    tpr_f1 = tpr[cutoff_index]
    #col_th = plt.cm.RdYlBu(np.linspace(0.05, 0.95, len(a)))

    
    if new:
        plt.subplots(1, 2, figsize = (12, 5))
    if colors is None:
        colors = ['darkorange', 'b']
    plt.subplot(121)
    plt.plot(fpr, tpr, color=colors[0], lw=2, zorder=0,
             label=f'ROC {version}, AUC= {roc_auc:0.2f}, F1={f1:0.2f}')
    # plt.scatter(fpr[cutoff_index], tpr[cutoff_index], color=colors[1], s=40, 
    #                 facecolors = 'none')
    plt.scatter(fpr_cutoff, tpr_cutoff, color='r', s=40, 
                facecolors = 'r', zorder = 1)
                #label=f'Cutoff {a:.2f} (FPR={fpr_cutoff:.2f}, TPR={tpr_cutoff:.2f})')
    plt.scatter(fpr_f1, tpr_f1, color='g', s=40, marker = 'x',
                facecolors = 'g', zorder = 1,
                label=f'Cutoff F1 {f1:.2f} (FPR={fpr_f1:.2f}, TPR={tpr_f1:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve {key}')
    #plt.legend([p2], [f'Prob {a}'], frameon = False)
    #plt.gca().add_artist(l1)
    plt.legend(loc="lower right")
    plt.subplot(122)
    plt.plot(rec, prec, color=colors[1], lw=2, zorder = 0,
             label=f'PR {version}, AP= {average_precision:0.2f}')
    plt.scatter(rec[index_pr], prec[index_pr], color='r', s=40, 
                facecolors = 'r', zorder = 1)
    plt.fill_between(rec, prec, alpha=0.2, color=colors[1])
    plt.xlabel('Recall TP/(TP+FN)') #how many of positives were correct
    plt.ylabel('Precision TP/(TP+FP)') #how many are actually positive
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall (PR) curve ({key})')
    plt.legend()
    if main_dir is not None:
        plt.savefig(f'{main_dir}/{key}_roc_pr_curves.png')
    
    

# plots confusion matrix
def confusion_matrix(true_labels, train_preds, version = '', key = '', main_dir = None):
    
    train_preds = np.hstack(train_preds)
    true_labels = np.hstack(true_labels)
    cm = metrics.confusion_matrix(true_labels, train_preds)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    accuracy = metrics.accuracy_score(true_labels, train_preds)
    precision = metrics.precision_score(true_labels, train_preds)
    recall = metrics.recall_score(true_labels, train_preds)
    f1 = metrics.f1_score(true_labels, train_preds)
    
    classes = ['Negative (0)', 'Positive (1)']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [3, 1]})
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, xticklabels=classes, yticklabels=classes)
    # Add text annotations for percentages
    for i in range(len(classes)):
            for j in range(len(classes)):
                color = 'black' if cm[i, j] < np.max(cm)*0.9 / 2 else 'white'
                ax1.text(j + 0.52, i + 0.62, '{:.1f}%'.format(cm_percentage[i, j]),
                        horizontalalignment='center', verticalalignment='center', 
                        color=color, weight='bold')
    ax1.set_xlabel('Predicted labels')
    ax1.set_ylabel('True labels')
    ax1.set_title(f'Confusion Matrix ({version}, {key})')
    # Add performance metrics to the right
    ax2.axis('off')
    ax2.text(0, 0.8, f'Accuracy: {accuracy:.2f}', fontsize=12, weight='bold', color='black')
    ax2.text(0, 0.6, f'Precision: {precision:.2f}', fontsize=12, weight='bold', color='black')
    ax2.text(0, 0.4, f'Recall: {recall:.2f}', fontsize=12, weight='bold', color='black')
    ax2.text(0, 0.2, f'F1-score: {f1:.2f}', fontsize=12, weight='bold', color='black')
    if main_dir is not None:
        plt.savefig(f'{main_dir}/{key}_conf_matrix.png')


# plotting probabilities by true positive and negative class
def broken_histogram(true_labels, proba, version, key = '', main_dir = None):
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
    ax1.set_title(f'classifier probability distributions\n{version}, {key}')
    if main_dir is not None:
        plt.savefig(f'{main_dir}/{key}_prob_hist_{version}.png')
        plt.close()



# plots true and false positive rates over thresholds with F1 score max cutoff
def f1_max_curves(true_labels, proba, version = '', key = '', main_dir = None):
    thresholds = np.linspace(0, 1)
    tpr_ = []; fnr_ = []
    f1 = []
    for z in thresholds:
        preds_cutoff = (proba > z).astype(int)
        f1.append(metrics.f1_score(true_labels, preds_cutoff))
        tp, fn, fp, tpr, fnr, fpr = scores(true_labels, preds_cutoff)
        tpr_.append(tpr); fnr_.append(fnr)
    plt.figure()
    plt.plot(thresholds, tpr_, label = 'TPR')
    plt.plot(thresholds, fnr_, label = 'FNR')
    plt.plot(thresholds, f1, label = 'f1-score')
    plt.scatter(thresholds[np.argmax(f1)], f1[np.argmax(f1)], color = 'r')
    plt.axvline(thresholds[np.argmax(f1)], ls = '--', color = 'r')
    plt.xlabel('thresholds'); plt.ylabel('rate');
    plt.legend()
    plt.title(f'metrics across probability cutoff thresholds\n{version}, {key}');
    if main_dir is not None:
        plt.savefig(f'{main_dir}/{key}_thresholds_{version}.png')
        plt.close()




def vizualize_dataloader(station, test_dfs, true_labels, train_preds, 
                         proba, main_dir = None):
    
    # mask_st = test_data.station == station
    # test_loader_df = [test_data.data[x] for x in np.where(mask_st)[0]]
    # mask_data = np.hstack(test_dfs['Code'] == station)
    # data = test_dfs[mask_data][:len(test_loader_df)]
    # time = test_dfs.index[mask_data][:len(test_loader_df)]
    
    n = train_preds.shape[0]
    mask_st = np.hstack(test_dfs['Code'] == station)
    data = test_dfs[mask_st][:n]
    time = test_dfs.index[mask_st][:n]
    label = data['Label'][:n]
    
    
    if (label == 1.).sum() > 0:
        event_idx = np.where(label == 1.)[0][0]
        disp = data[['N', 'E', 'U']].iloc[event_idx-1:event_idx+1]
        disp_delta = np.nanmax(np.abs(np.diff(disp, axis = 0))) / np.abs(disp.iloc[0,:])
    else:
        disp_delta = [0.]
    
    mask_st = mask_st[:n]
    mask_n = (np.in1d(train_preds[mask_st], [0, 1])) & (true_labels[mask_st] == 0)
    mask_tp = (train_preds[mask_st] == 1) & (true_labels[mask_st] == 1)
    mask_fn = (train_preds[mask_st] == 0) & (true_labels[mask_st] == 1)
    mask_fp = (train_preds[mask_st] == 1) & (true_labels[mask_st] == 0)
    
    # window_size = test_loader_df[0].shape[0]
    
    #plot the dataloader    
    #y = np.array([np.array(x[1]) for x in test_loader.dataset])
    fig, ax = plt.subplots(3, 1, figsize=(16, 6))
    
    plt.subplot(311)
    plot_helper(data, 'N', time, mask_n, mask_tp, mask_fn, mask_fp)
    plt.ylabel('N')
    plt.legend()
    
    plt.subplot(312)
    plot_helper(data, 'E', time, mask_n, mask_tp, mask_fn, mask_fp)
    plt.ylabel('E')
    #plt.legend()
    
    plt.subplot(313)
    plot_helper(data, 'U', time, mask_n, mask_tp, mask_fn, mask_fp)
    plt.ylabel('U')
    #plt.legend()
    
    plt.xlabel('time index')
    
    max_tp = np.nanmax(proba[mask_st][mask_tp]) if mask_tp.sum() >= 1 else np.nan
    max_fn = np.nanmax(proba[mask_st][mask_fn]) if mask_fn.sum() >= 1 else np.nan
    max_fp = np.nanmax(proba[mask_st][mask_fp]) if mask_fp.sum() >= 1 else np.nan
    text = f'max prob: TP = {max_tp: 0.2f}, FN = {max_fn: 0.2f}, FP = {max_fp: 0.2f}'
    # plt.text(x = 0.1, y = 0.9, s = text, transform = ax[-1].transAxes, 
    #          verticalalignment='top')
    plt.suptitle(f'station: {station}\n{text}, max displacement {np.max(disp_delta): 0.3f}')
    
    if main_dir is not None:
        plt.savefig(f'{main_dir}/dataloader_tp_fn.png')


def plot_helper(df, name, time, mask_n, mask_tp, mask_fn, mask_fp):
 
    time_formatted = to_datetime(time, format='%Y%m%d').values
    z = np.empty(len(time)); z[:] = np.nan
    z[mask_n] = df[name].loc[mask_n]
    plt.plot(time_formatted, df[name], color = '0.5')
    
    z = np.empty(len(time)); z[:] = np.nan
    z[mask_tp] = df[name].loc[mask_tp]
    plt.plot(time_formatted, z, '.-', color = 'g', alpha = 0.5, label = 'TP')
    
    z = np.empty(len(time)); z[:] = np.nan
    z[mask_fn] = df[name].loc[mask_fn]
    plt.plot(time_formatted, z, '.-', color = 'r', alpha = 0.5, label = 'FN')
    
    z = np.empty(len(time)); z[:] = np.nan
    z[mask_fp] = df[name].loc[mask_fp]
    plt.plot(time_formatted, z, '.-', color = 'b', alpha = 0.5, label = 'FP')

        
    
    
def plot_table(data, columns, versions, key = '', main_dir = None):

    
    # columns = ('TP', 'FN', 'FP')
    # columns = ('trp', 'fnr', 'fpr', 'acc')
    
    rows = [x for x in versions]

    # Get some pastel shades for the colors
    colors = plt.cm.BuPu(np.linspace(0.2, 0.7, len(rows)))
    n_rows = len(data)
    bar_width = (1./len(versions)) * 0.5
    
    x = np.arange(len(columns))
    multiplier = 0
    # Plot bars and create text labels for the table
    cell_text = []
    fig, (ax, tabax) = plt.subplots(nrows=2, figsize = (n_rows+1, 5),
                                    gridspec_kw={'height_ratios': [3, 1]})
    for row in range(n_rows):
        offset = bar_width * multiplier
        ax.bar(x+offset, data[row], bar_width, color=colors[row])
        #cell_text.append(['%1.1f' % (x / 1000.0) for x in y_offset])
        cell_text.append(['%.2f' % (x) for x in data[row]])
        multiplier += 1
    # Reverse colors and text labels to display the last value at the top.
    #colors = colors[::-1]
    #cell_text.reverse()
    ax.set_xticks([])
    ax.set_title(f'Metrics comparison, {key}')
    
    # Add a table at the bottom of the Axes
    tabax.axis("off")
    the_table = tabax.table(cellText=cell_text,
                          rowLabels=rows,
                          rowColours=colors,
                          colLabels=columns,
                          cellLoc='center',
                          loc='center')
    plt.subplots_adjust(left=0.2, bottom=0.01, hspace = 0.1)

    if main_dir is not None:
        plt.savefig(f'{main_dir}/{key}_metrics_compare.png', bbox_inches = 'tight')
        plt.close()
    
    
    



