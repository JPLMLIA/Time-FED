

#%%
from timefed.research.extract import roll

import pandas as pd

df = pd.read_hdf('timefed/research/tests/shampoo/shampoo.data.label.h5')
df



#%%

zero = pd.Timedelta(0)
freq = (df.index[1:] - df.index[:-1]).value_counts().sort_values(ascending=False)
freq

if not resolution:
    resolution   = freq[freq.keys() > zero].sort_index().index[0]

#%%

frequency = freq[freq.keys() > zero].sort_values(ascending=False).index[0]
frequency

window = '1 Y'
delta = pd.Timedelta(window)
size = int(delta / frequency)
size

frequency
delta

delta / pd.Timedelta('31 days')
