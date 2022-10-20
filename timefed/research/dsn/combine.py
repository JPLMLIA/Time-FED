

import h5py
import pandas as pd
from timefed.config import Config

config = Config('/data1/timefed/timefed/research/dsn/configs/tests/oc.process.mms.yml', 'process-default')
drs = pd.read_hdf(config.input.drs, 'MMS1')

#%%
def decode_strings(df):
    for name, column in df.items():
        if column.dtype == 'object':
            df[name] = column.apply(lambda string: string.decode())
    return df

def get_keys(file):
    keys = {}
    with h5py.File(file, 'r') as h5:
        for sc in h5.keys():
            tracks = keys[sc] = []
            for ant in h5[sc].keys():
                tracks += h5[f'{sc}/{ant}']
    return keys

keys = get_keys('/data1/timefed/local/process/mms/windows.h5')

drs = drs[['SCHEDULE_ITEM_ID', 'DR_CLOSURE_CAUSE_CD']]
drs = decode_strings(drs)

le = drs.query('DR_CLOSURE_CAUSE_CD == "LE"')
we = drs.query('DR_CLOSURE_CAUSE_CD == "WE"')
rfi = drs.query('DR_CLOSURE_CAUSE_CD == "RFI"')
dr = set(rfi.SCHEDULE_ITEM_ID)

tracks = set(keys['JWST1'])

keys = [float(k) for k in keys['JWST1']]
tracks = set(keys)

s1, s2, s3, s4 = set(keys['MMS1']), set(keys['MMS2']), set(keys['MMS3']), set(keys['MMS4'])
tracks = set.union(s1, s2, s3, s4)

avail = tracks.intersection(dr)

#%%
def windows(file):
    def get_keys(file):
        """
        """
        keys = []
        with h5py.File(file, 'r') as h5:
            for sc in h5.keys():
                for ant in h5[sc].keys():
                    for track in h5[sc][ant].keys():
                        for dcc in h5[sc][ant][track].keys():
                            keys.append(f'{sc}/{ant}/{track}/{dcc}')
        return keys
    keys = get_keys(file)
    pos = 0
    neg = 0
    for k in keys:
        df = pd.read_hdf(file, k)
        pos += df.query('Label == 1').shape[0]
        neg += df.query('Label == 0').shape[0]
    return pos, neg

input = f'{config.output.windows}.0'
paths = get_keys(avail, input)

dfs = []
for path in paths:
    dfs.append(pd.read_hdf(config.output.windows, path))

df = pd.concat(dfs)
df = df.sort_index()
pf = nf.query('Label == 1')

#%%

def split(df, percent):
    split = int(df.shape[0] * percent)
    date  = df.iloc[split].name
    train = df.query('index <= @date')
    test  = df.query('index > @date')
    print(f'Date: {date}')
    print('Train:')
    print(f'- Total = {train.shape[0]}')
    print(f'- Perc% = {(train.shape[0] / df.shape[0]) * 100:.2f}')
    print('Test:')
    print(f'- Total = {test.shape[0]}')
    print(f'- Perc% = {(test.shape[0] / df.shape[0]) * 100:.2f}')
    return date
date = split(pf, .8)
# RFI: 2022-04-27 17:02:07

#%%

def get_keys_neg(tracks, file):
    """
    """
    keys = []
    with h5py.File(file, 'r') as h5:
        for sc in h5.keys():
            for ant in h5[sc].keys():
                for track in h5[sc][ant].keys():
                    if float(track) in tracks:
                        continue
                    for dcc in h5[sc][ant][track].keys():
                        keys.append(f'{sc}/{ant}/{track}/{dcc}')
    return keys

def get_keys_flat(file):
    """
    """
    keys = []
    with h5py.File(file, 'r') as h5:
        for sc in h5.keys():
            for ant in h5[sc].keys():
                for track in h5[sc][ant].keys():
                    for dcc in h5[sc][ant][track].keys():
                        keys.append(f'{sc}/{ant}/{track}/{dcc}')
    return keys

#%%
from tqdm import tqdm
start, end = pf.iloc[0].name, pf.iloc[-1].name

negs = {}
tfs = []
for key in tqdm(keys[:int(len(keys)/2)]):
    tf = pd.read_hdf(config.output.windows, key)
    if tf.Label.any():
        continue
    pass# tf = tf.sort_index()
    pass# tf = tf.query('@start <= index')
    pass# tf = tf.query('index <= @end')
    pass#tf = tf.query('(@start <= index) and (index <= @end)')
    if not tf.empty:
        negs[key] = tf.shape[0]
        tfs.append(tf)

total = 0
accept = []
for key, n in negs.items():
    if total < pf.shape[0]:
        total += n
        accept.append(key)

dfs = []
for key in accept:
    dfs.append(pd.read_hdf(config.output.windows, key))
df = pd.concat(dfs)
df = df.sort_index()

df.query('index <= @date')


39696
9922

#%%
any([*[f in col for f in ['diff_AGC_VOLTAGE', 'diff_CARRIER_SYSTEM_NOISE_TEMP']], not col.startswith('diff')])
col = 'diff_CARRIER_SYSTEM_NOISE_TEMP'

#%%
#%%
#%%

import itertools


d = {'a': [1], 'b': [1, 2]}

e = [set(d[a]) for a in d]

list(itertools.chain(e))



s1, s2 = set(d['a']), set(d['b'])
