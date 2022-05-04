#%%
%load_ext autoreload
%autoreload 2
%matplotlib inline
#%%
import pandas as pd

train = pd.read_hdf('/Users/jamesmo/projects/mloc/timefed/research/dsn/local/data/features.h5', 'MMS1/train')
test  = pd.read_hdf('/Users/jamesmo/projects/mloc/timefed/research/dsn/local/data/features.h5', 'MMS1/test')

import numpy as np

inf_cols = train.columns[np.isinf(train).any()]
train = train.drop(columns=inf_cols)
test  = test .drop(columns=inf_cols)

#%%
# Percent of pos/neg rows
train['Label'].value_counts() / train['Label'].count() * 100
test['Label'].value_counts() / test['Label'].count() * 100

#%%

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score
)

#%%

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(train.drop(columns=['Label']), train['Label'])

pred = model.predict(test.drop(columns=['Label']))

truth = test['Label']
accuracy_score(truth, pred)
precision_score(truth, pred)
recall_score(truth, pred)
roc_auc_score(truth, pred)

cm = confusion_matrix(truth, pred)

#%%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(20, 20))

ax.set_title('Confusion Matrix')
ax.plot(cm)

#%%

from sklearn.metrics import ConfusionMatrixDisplay

# fig, ax = plt.subplots(1, 1, figsize=(20, 20))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

#%%

from sklearn.metrics import classification_report

names = ['DR Negative', 'DR Positive']

print(classification_report(truth, pred, target_names=names))
