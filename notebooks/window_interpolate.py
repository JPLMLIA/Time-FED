"""
Interpolates over windows to measure the performance of a given interpolation
method against true data.

Example usage script:

#!/bin/bash
shopt -s expand_aliases

conda activate mloc

alias pywi='python window_interpolate.py -f data.h5 -ki merged -o results.h5 -m 1 2 3 5 10 15 -k linear cubic quadratic'
#alias pywi='python window_interpolate.py -f data.h5 -ki merged -o results.h5 -t 60 -m 60 -k linear cubic quadratic'

pywi -b 2018-04-01-20:00 2018-04-02-08:00 -ko spring
pywi -b 2018-07-01-20:00 2018-07-02-08:00 -ko summer
pywi -b 2018-10-01-20:00 2018-10-02-08:00 -ko fall
pywi -b 2018-12-01-20:00 2018-12-02-08:00 -ko winter

"""
import argparse
import multiprocessing as mp
import numpy  as np
import pandas as pd
import warnings

from scipy.interpolate import interp1d
from tqdm              import tqdm

warnings.filterwarnings('ignore')

def predict(data, i, m, t, kind):
    testing  = data.iloc[i:i+m]

    assert testing.isnull().mean() < .25

    t1 = data.iloc[i-t : i]
    t2 = data.iloc[i+m : i+m+t]

    assert t1.isnull().mean() < .25
    assert t2.isnull().mean() < .25

    training = pd.concat([t1, t2]).dropna()

    interp  = interp1d(x=training.index, y=training.values, kind=kind)
    predict = interp  (x=testing.index)

    # diff = testing - predict
    # diff = np.abs(testing - predict) / ((testing + predict) / 2) * 100
    diff = (np.abs(testing - predict) / testing) * 100

    assert diff.any()

    return diff

def measure(flags):
    col, m, t, kind = flags

    data  = df[col]
    flat  = []
    wins  = 0

    for i in range(t, data.shape[0]-m-t):
        try:
            diff = predict(data, i, m, t, kind)
        except:
            continue

        if diff.any():
            flat += list(diff)
            wins += 1

        # This limits the total number of interpolated windows to gather metrics from
        if args.n_windows is not None:
            if wins == args.n_windows:
                break

    mean = np.nanmean(flat)
    std  = np.nanstd(flat, ddof=1)
    sqrd = np.nanmean(np.array(flat)**2)

    return col, m, t, kind, wins, mean, std, sqrd

#%%

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--file',         type=str,
                                                required=True,
                                                help='Path to h5 containing a pandas dataframe'
    )
    parser.add_argument('-ki', '--key_in',      type=str,
                                                default='data'                                          ,
                                                help='The key to use when reading the --file'
    )
    parser.add_argument('-c', '--columns',      type=str,
                                                nargs='+',
                                                help='Selects which columns to use. Leave empty for all'
    )
    parser.add_argument('-m', '--windows',      type=int,
                                                nargs='+',
                                                default=[30, 60, 90, 120],
                                                help='Window sizes to use'
    )
    parser.add_argument('-t', '--trainings',    type=int,
                                                nargs='+',
                                                default=[5, 15, 30, 45, 60, 75, 90],
                                                help='Training sizes to use'
    )
    parser.add_argument('-k', '--kinds',        type=str,
                                                nargs='+',
                                                default=['linear', 'nearest', 'next', 'previous', 'cubic', 'quadratic'],
                                                help='Interpolation kinds to use'
    )
    parser.add_argument('-b', '--between',      type=str,
                                                nargs=2,
                                                help='Subselects between [datetime1, datetime2)'
    )
    parser.add_argument('-o', '--out',          type=str,
                                                required=True,
                                                help='Path to h5 containing a pandas dataframe'
    )
    parser.add_argument('-ko', '--key_out',     type=str,
                                                default='data',
                                                help='The key to use when writing to --out'
    )
    parser.add_argument('-nw', '--n_windows',   type=int,
                                                default=10,
                                                help='Number of windows to roll for'
    )

    args = parser.parse_args()

    # Load the data
    df = pd.read_hdf(args.file, args.key_in)

    # Ensure the index is integers
    if df.index.dtype != 'int64':
        df = df.reset_index()

    # Subselect between dates if given
    if args.between:
        lower, upper = args.between
        df = df[(lower <= df.datetime) & (df.datetime < upper)]

    # Drop the datetime column if it's present
    if 'datetime' in df:
        df = df.drop(columns=['datetime'])

    # Drop other columns if not requested
    if args.columns:
        df = df.drop(columns=list(set(df.columns)-set(args.columns)))

    # Test combinations
    m = args.windows
    t = args.trainings
    k = args.kinds

    metrics = ['mean', 'std', 'sqrd']

    # Setup results dataframe
    index  = [(_m, _t, _k) for _m in m for _t in t for _k in k]
    # header = [(c, 'windows', *metrics) for c in df]
    nf = pd.DataFrame(columns=pd.MultiIndex.from_product([df.columns, metrics+['windows']]), index=pd.MultiIndex.from_tuples(index))
    nf.index.names = ['m', 't', 'kind']

    # Create argument combinations
    combos = [(c, _m, _t, _k) for c in df for _m in m for _t in t for _k in k]

    with mp.Pool() as pool:
        for c, m, t, k, wins, mean, std, sqrd in tqdm(pool.imap_unordered(measure, combos), total=len(combos), desc='Combos'):
            nf[c, 'mean'][m, t, k] = mean
            nf[c, 'std' ][m, t, k] = std
            nf[c, 'sqrd'][m, t, k] = sqrd
            nf[c, 'windows'][m, t, k] = wins

    nf.to_hdf(args.out, args.key_out)

#%%
