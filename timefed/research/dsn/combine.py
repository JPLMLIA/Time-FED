

import h5py
import pandas as pd
from timefed.config import Config

config = Config('timefed/research/dsn/configs/tests/oc.process.mro.le.yml', 'process-default')
drs = pd.read_hdf(config.input.drs, 'MRO')

def get_keys(file):
    keys = {}
    with h5py.File(file, 'r') as h5:
        for sc in h5.keys():
            tracks = keys[sc] = []
            for ant in h5[sc].keys():
                tracks += h5[f'{sc}/{ant}']
    return keys

keys = get_keys(config.output.windows)

drs = drs[['SCHEDULE_ITEM_ID', 'DR_CLOSURE_CAUSE_CD']]

we = drs.query('SCHEDULE_ITEM_ID == "LE"')
dr = set(we.SCHEDULE_ITEM_ID)

tracks = set(keys['MRO'])

avail = tracks.intersection(dr)

tracks


s = set()
dir(s)
