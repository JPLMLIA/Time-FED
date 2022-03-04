tf = tf.set_index('RECEIVED_AT_TS')

#%%
import datetime as dt


dtt.fromordinal(tf.index[0])
tf.index[0]
dtt.fromtimestamp(tf.index[0])
dtt.fromtimestamp(tf.index[1])
dtt.fromtimestamp(tf.index[2])


tf.index

pd.DatetimeIndex(tf.index)

dtt.fromtimestamp(tf.index)

nf.index = [dtt.fromtimestamp(ts) for ts in tf.index]

#%%
from datetime import datetime as dtt

def timestamp_to_datetime(timestamps):
    """
    Converts an integer or floating point timestamp to a Python datetime object.

    Parameters
    ----------
    timestamps: single or iterable of int or float
        A singular or a list of timestamps to convert

    Returns
    -------
    list of or single datetime
    """
    if isinstance(timestamps, (int, float)):
        return dtt.fromtimestamp(timestamps)
    else:
        return [dtt.fromtimestamp(ts) for ts in timestamps]

nf = tf.copy()
nf.index = timestamp_to_datetime(nf.index)
#%%

for name, column in nf.items():
    if 'DT' in name:
        nf[name] = timestamp_to_datetime(column)

#%%

def decode_strings(df):
    """
    Attempts to apply string.decode() to any column with a dtype of object.

    Parameters
    ----------
    df: pandas.DataFrame
        The DataFrame object to iterate over searching for object dtype columns
        to apply decoding to

    Returns
    -------
    df: pandas.DataFrame
        Same DataFrame object as input but with decoded columns
    """
    for name, column in nf.items():
        if column.dtype == 'object':
            try:
                df[name] = column.apply(lambda string: string.decode())
                print(f'Decoded column {name}')
            except:
                print(f'Failed to decode column {name}')

    return df

nf = decode_strings(nf)
