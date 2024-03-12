"""
Experimental preprocessing for DSN tracks to prepare the data for the TimeFED pipeline.
"""
import argparse
import h5py
import logging
import numpy  as np
import pandas as pd
import warnings

from datetime import datetime as dtt
from mlky     import Config
from tables   import NaturalNameWarning
from tqdm     import tqdm

from timefed import utils

# Disable tables warnings
warnings.filterwarnings('ignore', category=NaturalNameWarning)

# Disable pandas warnings
pd.options.mode.chained_assignment = None

Logger = logging.getLogger('timefed/preprocess/dsn.py')


def subsample(df):
    """
    Subsamples positive and negative tracks to balance the classes per the config

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame containing positive/negative tracks to subsample from

    Returns
    -------
    df: pandas.DataFrame
        DataFrame with randomly selected subsample
    """
    # Section of config for this function
    config = Config.preprocess.subsample

    # If a query string is given, apply it first
    if config.query:
        Logger.info(f'Applying query: {config.query}')
        df = df.query(config.query)

    # Group by SID, taking the mean value of the Label to differentiate between positive and negative tracks (positive has mean value > 0, neg == 0)
    gf  = df[['SCHEDULE_ITEM_ID', 'Label']].groupby('SCHEDULE_ITEM_ID').mean()
    pos = gf.query('Label  > 0')
    neg = gf.query('Label == 0')

    # Subsample the positive tracks using a percent of the total
    pos_tracks = pos.index
    if isinstance(config.pos_limit, (int, float)):
        # Int takes N many tracks
        if isinstance(config.pos_limit, int):
            count = config.pos_limit
        # Float takes a fraction of the total of positive tracks
        else:
            count = int(pos.shape[0] * config.pos_limit)

        # Randomly select from the available tracks
        pos_tracks = np.random.choice(pos.index, count, replace=False)
        Logger.info(f'Randomly selecting {count}/{pos.shape[0]} ({count/pos.shape[0]*100:.2f}%) positive tracks')

    # Subsample the negative tracks using a ratio to the number of positive
    neg_tracks = neg.index
    if config.neg_ratio:
        ratio = int(len(pos_tracks) * config.neg_ratio)
        neg_tracks = np.random.choice(neg.index, ratio, replace=False)
        Logger.info(f'Randomly selecting {ratio}/{neg.shape[0]} ({ratio/neg.shape[0]*100:.2f}%) negative tracks')

    # Use neg_limit to select negative tracks
    if isinstance(config.neg_limit, (int, float)):
        # Int takes N many tracks
        if isinstance(config.neg_limit, int):
            count = config.neg_limit
        # Float takes a fraction of the total of negative tracks
        else:
            count = int(neg.shape[0] * config.neg_limit)

        # Randomly select from the available tracks
        neg_tracks = np.random.choice(neg.index, count, replace=False)
        Logger.info(f'Randomly selecting {count}/{neg.shape[0]} ({count/neg.shape[0]*100:.2f}%) negative tracks')

    # Now select those tracks plus the positive ones from the df
    tracks = list(pos_tracks) + list(neg_tracks)
    df = df.query('SCHEDULE_ITEM_ID in @tracks')

    # Report stats
    gf  = df[['SCHEDULE_ITEM_ID', 'Label']].groupby('SCHEDULE_ITEM_ID').mean()
    pos = gf.query('Label  > 0')
    neg = gf.query('Label == 0')

    Logger.info(f'Number of tracks total  : {gf.shape[0]:5}')
    Logger.info(f'Number that are positive: {pos.shape[0]:5} ({pos.shape[0]/gf.shape[0]*100:.2f}%)')
    Logger.info(f'Number that are negative: {neg.shape[0]:5} ({neg.shape[0]/gf.shape[0]*100:.2f}%)')

    return df


def analyze(df):
    """
    Analyzes the final dataframe and removes invalid labels

    Parameters
    ----------
    df: pandas.DataFrame
        The dataframe of concatenated track frames

    Returns
    -------
    df: pandas.DataFrame
        The same dataframe but with the invalid class removed
    """
    size   = df.shape[0]
    counts = df.Label.value_counts()
    tracks = df.SCHEDULE_ITEM_ID.value_counts().size

    Logger.info('Stats on the combined dataframe:')
    Logger.info(f'Number of tracks   : {tracks}')
    Logger.info(f'Total timestamps   : {size}')
    Logger.info(f'Percent Label is  1: {counts[1]/size*100:.2f}%')
    Logger.info(f'Percent Label is  0: {counts[0]/size*100:.2f}%')
    Logger.info(f'Percent Label is -1: {counts[-1]/size*100:.2f}%')

    df = df.query('Label != -1')

    Logger.info(f'Removing invalid labels (-1) reduced the data by {(1-df.shape[0]/size)*100:.2f}%')

    size    = df.shape[0]
    counts  = df.Label.value_counts()
    ntracks = df.SCHEDULE_ITEM_ID.value_counts().size

    Logger.info(f'This removed {tracks - ntracks} ({(1-ntracks/tracks)*100:.2f}%) tracks, leaving {ntracks}')
    Logger.info(f'Total timestamps   : {size}')
    Logger.info(f'Percent Label is  1: {counts[1]/size*100:.2f}%')
    Logger.info(f'Percent Label is  0: {counts[0]/size*100:.2f}%')

    Logger.info('Dropping rows with a NaN in any column')

    df      = df.dropna(how='any', axis=0)
    tracks  = ntracks
    ntracks = df.SCHEDULE_ITEM_ID.value_counts().size

    Logger.info(f'Total rows removed: {size-df.shape[0]} ({(1-df.shape[0]/size)*100:.2f}%)')
    Logger.info(f'This removed {tracks - ntracks} ({(1-ntracks/tracks)*100:.2f}%) tracks, leaving {ntracks}')

    gf  = df[['SCHEDULE_ITEM_ID', 'Label']].groupby('SCHEDULE_ITEM_ID').mean()
    vc  = (gf != 0).value_counts()
    sum = vc.sum()

    Logger.info(f'Number of tracks with positive labels  : {vc[True]:5} ({vc[True]/sum*100:.2f}%)')
    Logger.info(f'Number of tracks that are only negative: {vc[False]:5} ({vc[False]/sum*100:.2f}%)')

    return df


def add_features(df):
    """
    Adds additional features to a track's dataframe per the Config.preprocess.

    Parameters
    ----------
    df: pandas.DataFrame
        Track DataFrame to add a columns to

    Returns
    -------
    df: pandas.DataFrame
        Modified track DataFrame
    """
    for feature in Config.preprocess.features.diff:
        if feature not in df:
            Logger.error(f'Feature not in DataFrame: {feature}')
            continue
        df[f'diff_{feature}'] = df[feature].diff()

    return df


def add_label(df, drs):
    """
    Adds the `Label` column to the input DataFrame and marks timestamps as:
         0: Negative class (no DR)
         1: Positive class (had DR)
        -1:  Invalid class (Bad frames)

    Parameters
    ----------
    df: pandas.DataFrame
        Track DataFrame to add a Label column to
    drs: pandas.DataFrame
        The DataFrame containing DR information

    Returns
    -------
    df: pandas.DataFrame
        Modified track DataFrame
    """
    # Create label column with -1 as "bad" rows
    df['Label'] = -1

    # Set rows between B/EoT as 0
    df.Label.loc[df.query('BEGINNING_OF_TRACK_TIME_DT <= RECEIVED_AT_TS <= END_OF_TRACK_TIME_DT').index] = 0

    # Exclude bad frames from the negative class
    df.Label.loc[df.query('TLM_BAD_FRAME_COUNT > 0').index] = -1

    # Verify the bad frames were removed correctly
    assert df.query('Label == 0 and TLM_BAD_FRAME_COUNT > 0').empty, 'Failed to remove bad frames from negative class'

    # Lookup if this track had a DR, if so change those timestamps to 1
    lookup = drs.query(f'SCHEDULE_ITEM_ID == {df.SCHEDULE_ITEM_ID.iloc[0]}')
    if not lookup.empty:
        incident = *timestamp_to_datetime(lookup.INCIDENT_START_TIME_DT), *timestamp_to_datetime(lookup.INCIDENT_END_TIME_DT)

        # Set the incident as positive
        df.Label.loc[df.query('@incident[0] <= RECEIVED_AT_TS <= @incident[1]').index] = 1

    return df


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
    for name, column in df.items():
        if column.dtype == 'object':
            try:
                df[name] = column.apply(lambda string: string.decode())
                # Logger.debug(f'Decoded column {name}')
            except:
                Logger.exception(f'Failed to decode column {name}')

    return df


def get_keys(file):
    """
    Retrieves the keys of the input h5 file

    Parameters
    ----------
    file: str
        Path to h5 file to read

    Returns
    -------
    keys: dict
        Dictionary of keys in the h5
    """
    keys = {}
    with h5py.File(file, 'r') as h5:
        for mission in h5.keys():
            keys[mission] = {}
            for ant in h5[mission].keys():
                keys[mission][ant] = list(h5[f'{mission}/{ant}'].keys())

    return keys


def preprocess(mission, keys):
    """
    Preprocesses all the tracks for a given mission to prepare the data for the
    pipeline scripts.

    Parameters
    ----------
    mission: str
        The string name of the mission being processed
    keys: dict
        Dictionary of {antenna: [tracks]} for this mission
    """
    Logger.info(f'Processing tracks for mission: {mission}')

    # Retrieve the DRs for this mission
    try:
        Logger.info('Retrieving DRs')
        drs = pd.read_hdf(Config.preprocess.drs, mission)
        drs = decode_strings(drs)
    except:
        Logger.exception(f'Failed to retrieve DRs for mission {mission} from file {Config.preprocess.drs}, returning early')
        return False

    # Skip tracks that have wrong DRs
    skip = []
    if Config.preprocess.only.drs:
        skip = list(drs.query(f'DR_CLOSURE_CAUSE_CD not in {Config.preprocess.only.drs}').SCHEDULE_ITEM_ID.astype(str))
        Logger.debug(f'Processing only DRs: {Config.preprocess.only.drs}')
        Logger.info(f'{len(skip)} tracks will be skipped due to being the wrong DR')

    # Setup TQDM
    total = sum([len(tracks) for _, tracks in keys.items()])
    bar   = tqdm(total=total, desc=f'Processing {mission} tracks')

    # List to store track frames
    dfs = []

    # Iterate over all antennae, tracks for this mission
    for ant, tracks in keys.items():
        for track in tracks:
            if track in ['-1.0']:
                bar.update()
                Logger.info(f'Skipping bad track ID: {track}')
                continue

            if track in skip:
                bar.update()
                continue

            # Load this track in
            key = f'{mission}/{ant}/{track}'

            # Check if there are DCC channels
            with h5py.File(Config.preprocess.tracks, 'r') as h5:
                dccs = list(h5[key].keys())

            if len(dccs) > 1:
                Logger.debug(f'Track {track} has {len(dccs)}: {dccs}')
                if Config.preprocess.skip_dcc:
                    dccs = []

            for dcc in dccs:
                subkey = f'{key}/{dcc}'
                df  = pd.read_hdf(Config.preprocess.tracks, subkey)

                # Decode the strings to cleanup column values (removes the b'')
                df = decode_strings(df)

                # Next attempt to convert DT and TS columns to python DT objects
                for name, column in df.items():
                    if 'DT' in name or 'TS' in name:
                        try:
                            df[name] = timestamp_to_datetime(column)
                        except:
                            Logger.exception(f'Failed to convert {name} to datetime for {subkey}, skipping track')
                            continue

                try:
                    # Create the label column
                    df = add_label(df, drs)
                except:
                    Logger.exception(f'Failed to add label to {subkey}, skipping track')
                    raise

                # Compute additional features
                df = add_features(df)

                # Set the index before saving
                df = df.set_index('RECEIVED_AT_TS')
                df.index.name = 'datetime'

                # Save to output and update TQDM
                df.to_hdf(Config.preprocess.file, f'preprocess/{subkey}')

                dfs.append(df)

            # Logger.debug(f'Successfully processed {subkey}')
            bar.update()

    Logger.info('Concatenating all frames together')
    df = pd.concat(dfs)
    df = analyze(df)

    if Config.preprocess.subsample:
        Logger.info('Subsampling tracks')
        df = subsample(df)

    df.to_hdf(Config.preprocess.file, f'preprocess/{mission}/complete')

    return True

def main():
    """
    Sends each missions' tracks off to preprocessing

    Returns
    -------
    True or None
        Whether the function processed each mission successfully

    Notes
    -----
    config keys:
        input:
            tracks: str
        only:
            missions: list of str
    """
    # Retrieve key structure of the tracks h5
    keys = get_keys(Config.preprocess.tracks)

    # Subselect which missions to process
    missions = Config.preprocess.only.missions
    if not missions:
        missions = list(keys.keys())

    skip = set()
    for mission in missions:
        if mission not in keys:
            Logger.warning(f'Mission {mission} not found in tracks h5, skipping')
            skip.update([mission])
            continue

        if not preprocess(mission, keys[mission]):
            skip.update([mission])

    if Config.preprocess.concat:
        Logger.info('Concatenating mission frames together')
        data = []
        for mission in set(missions) - skip:
            data.append(
                pd.read_hdf(Config.preprocess.file, f'preprocess/{mission}/complete')
            )
        df = pd.concat(data)
        df.to_hdf(Config.preprocess.file, 'preprocess/complete')

    return True


if __name__ == '__main__':
    # Initialize the loggers
    utils.init(args)

    # Process
    try:
        state = main()
    except Exception:
        Logger.exception('Caught an exception during runtime')
    finally:
        if state is not None:
            Logger.info('Finished successfully')
        else:
            Logger.info('Failed to complete')
