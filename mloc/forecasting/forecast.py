"""
Author: James Montgomery
Date  : September 1st, 2021
"""
import argparse
import logging
import pandas as pd
import numpy  as np
import os
import sys
import tsfresh

from collections         import Counter
from pvlib.solarposition import get_solarposition
from tqdm import tqdm
from glob import glob

# from tsfresh.utilities.dataframe_functions import impute

from mloc import utils


# Maps the case name to the actual column name in the data
Names = {
    'r0' : 'r0_10T',
    'Cn2': 'Cn2_10T',
    'pwv': 'water_vapor',
    'temperature'      : 'temperature',
    'pressure'         : 'pressure',
    'relative_humidity': 'relative_humidity',
    'wind_speed'       : 'wind_speed'
}

# The resolution of the data (in minutes) of each case
Resolution = {
    'r0' : 5,
    'pwv': 30,
    'temperature'      : 5,
    'pressure'         : 5,
    'relative_humidity': 5,
    'wind_speed'       : 5
}


def roll(df, case):
    """
    Rolls over a DataFrame and creates viable windows of the data to be yielded.
    The viability of a window depends on the case, see the notes section for
    more details.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame to roll over
    case: str
        The case of this run

    Yields
    ------
    pandas.DataFrame

    Notes
    -----
    Window viability requirements for each case:
    - r0 has 4 runs, a viable window must satisfy the least restrictive run, ie.
      weather only: temperature, pressure, wind speed, and humidity
    - PWV has 1 run, a viable window must have all variables for this run, ie.
      temperature, pressure, wind speed, humidity, and dewpoint

    Assumptions for each case:
    - r0
      - Window size is 50 minutes
      - Sampling rate of the data is 5 minutes
      - 50/5 = 10 samples per window
    - PWV
      - Window size is 300 minutes
      - Sampling rate of the data is 30 minutes
      - 300/30 = 10 samples per window

    Windows are considered viable if:
    - The minimum required variables are present in the window
      - No NaNs in any required variable
    - The window size is equal to the requirements for the case

    """
    # Window sizes for each case
    window = {
        'r0' : 50,
        'pwv': 300,
        'temperature'      : 50,
        'pressure'         : 50,
        'relative_humidity': 50,
        'wind_speed'       : 50
    }
    # Minimum columns for a viable window
    viable = {
        'r0' : ['temperature', 'pressure', 'wind_speed', 'relative_humidity'],
        'pwv': ['temperature', 'pressure', 'wind_speed', 'humidity', 'dewpoint'],
        'temperature'      : ['temperature', 'pressure','relative_humidity', 'wind_speed'],
        'pressure'         : ['temperature', 'pressure','relative_humidity', 'wind_speed'],
        'relative_humidity': ['temperature', 'pressure','relative_humidity', 'wind_speed'],
        'wind_speed'       : ['temperature', 'pressure','relative_humidity', 'wind_speed'],
    }

    # Calculate the integer distance for the window size of this case
    distance = int(window[case]/Resolution[case])

    delta = pd.Timedelta(f'{window[case]} min')
    count = Counter()
    valid = set()
    for i in range(0, df.index.size - window[case]):
        # Determine a window and check if it's the expected size
        j = i + distance
        if df.index[j] - df.index[i] > delta:
            continue

        sub = df.iloc[i:j]

        # Drop columns that have a NaN
        sub = sub.dropna(how='any', axis=1)

        # Verify this window has the minimum required columns
        if all([column in sub for column in viable[case]]):
            count.update(sub)
            valid.add((i, j))

    if len(valid) == 0:
        logger.debug('There were no valid windows, returning')
        return []

    logger.debug(f'There are {len(valid)} valid windows. The number of windows each key is in:')
    fmt = max([len(c) for c in count])
    for key, n in count.items():
        logger.debug(f"{key}{' '*(fmt-len(key))}: {n}")

    # Now yield the windows
    for i, j in tqdm(valid, desc='Windows'):
        yield df.iloc[i:j].dropna(how='any', axis=1)

def calculate_features(df):
    """
    Calculates new features

    Parameters
    ----------
    df: pandas.DataFrame
        The dataframe to calculate new features on

    Returns
    -------
    df: pandas.DataFrame
        The same input frame but with additional calculated features
    """
    # Always override the SZA feature (or add it)
    df['solar_zenith_angle'] = get_solarposition(
        time      = df.index,
        latitude  = 34.380000000000003,
        longitude = -1.176800000000000e+02,
        altitude  = 2280
    ).zenith

    # Always add day and minute
    df['day']    = df.index.dayofyear
    df['minute'] = df.index.hour * 60 + df.index.minute

    # if Cn2 exists, log it and drop the original
    if Names['Cn2'] in df:
        logger.debug('Created log(Cn2) column')
        df['log_Cn2_10T'] = np.log10(df['Cn2_10T'])
        df = df.drop(columns=['Cn2_10T'])

    return df

def extract(df):
    """
    Performs the tsfresh.extract_features() function on an input DataFrame. All
    tsfresh features are calculated with their default parameters and imputation
    is done on the output features frame. The last value of the index of the
    input data will be the index of the resulting features frame, where the
    features frame is 1 sample by N features.

    Parameters
    ----------
    df: pandas.DataFrame
        A DataFrame to pass to tsfresh.extract_features(). Assumes:
        - There is no class ID for the data (column_id of tsfresh)
        - The time array is the index of the frame

    Returns
    -------
    extract: pandas.DataFrame
        A DataFrame of 1 samples by N features. The index of this frame is the
        last index of the input frame.
    """
    df['_ID']   = np.full(len(df), 0)
    df['_TIME'] = df.index
    extract = tsfresh.extract_features(
        df,
        column_id    = '_ID',
        column_sort  = '_TIME',
        column_kind  = None,
        column_value = None,
        # impute_function     = impute,
        disable_progressbar = True,
        n_jobs = 1
    )

    # Imitate the original index
    extract.index = [df.index[-1]]

    # Add back in the original columns
    extract[df.columns] = df.loc[extract.index]

    # Drop the additional columns created for tsfresh
    extract = extract.drop(columns=['_ID', '_TIME'])

    return extract

def extract_features(df, case):
    """
    Extracts tsfresh features on rolling windows of the data and concatenates
    the feature frames together to produce a single frame for forecasting with.

    Parameters
    ----------
    df: pandas.DateFrame
        The data to generate rolling windows on and feature extract
    case: string
        The case being processed

    Returns
    -------
    ret: pandas.DataFrame
        The concatenated and time-sorted frame post-feature extraction
    """
    # Create the historical column if it is available, shift by the resolution
    if Names[case] in df:
        # df = df.rename(columns={Names[case]: f'historical_feature_{Names[case]}'})
        df[f'historical_feature_{Names[case]}'] = df[Names[case]].shift(freq=f'{Resolution[case]} min')
        df = df.drop(columns=[Names[case]])
        logger.debug(f'Created historical column of {Names[case]}')

    logger.debug(f'Percent of NaNs in each column:\n{(df.isna().sum() / df.index.size) * 100}')

    # Create windows and extract features
    logger.info('Creating rolling windows and extracting features')
    windows  = roll(df, case)
    extracts = []
    with utils.Pool() as pool:
        for ret in pool.imap_unordered(extract, windows, chunksize=100):
            extracts.append(ret)

    if not extracts:
        logger.debug('No extractions retrieved, returning')
        return False

    # Combine the extracted features back into a DataFrame
    logger.info('Concatting the feature frames together')
    ret = pd.concat(extracts)
    ret.sort_index(inplace=True)

    # Only keep features that will be used later
    feats = utils.load_pkl(os.path.join(DEPLOYDIR, case, 'features', 'all_features.pkl'))
    ret   = ret.drop(columns=set(ret.columns) - set(feats), errors='ignore')

    # Save the windows for future reference
    ret.to_hdf(os.path.join(DEPLOYDIR, case, '_data', 'windows.h5'), 'windows')

    # Save the last timestamp to know where to begin again
    utils.save_pkl(os.path.join(DEPLOYDIR, case, '_data', 'last_window.pkl'), ret.index[-1])

    return ret

def forecast(case, run, df, forecasts, cadence):
    """
    Produces forecasts for a given case run.

    Parameters
    ----------
    case: string
        The case being processed
    run: string
        The run for this case
    df: pandas.DataFrame
        The data to be forecasted with
    forecasts: int
        How far into the future to forecast
    cadence: int
        How often to forecast

    Notes
    -----
    Saves the forecasts to a CSV per run per case. This file will be appended to
    if new forecasts are generated.

    Forecasting and cadence:
        - Forecasting is how far to forecast. This value should be
          N * cadence <= [r0:180|pwv:300]. Examples:
            - r0 : With cadence =  5, to forecast 3 hours do 180/5  = 36
            - r0 : With cadence = 10, to forecast 3 hours do 180/10 = 18
            - pwv: With cadence = 30, to forecast 3 hours do 180/30 = 6
            - pwv: With cadence = 60, to forecast 3 hours do 180/60 = 2
          Defaults to 3 hours for any cadence in any case
        - Cadence is how often to forecast up to the forecast distance. This
          value must be a multiple of the resolution for the case. Examples:
            - r0 must be a multiple of 5
            - PWV must be a multiple of 30
          Defaults to the resolution of the case if not set
    """
    if df.isna().any().any():
        logger.debug(f'Percent of NaNs in columns that had NaNs:\n{(df[df.columns[df.isna().any()]].isna().sum() / df.index.size) * 100}')

    # Get the models to be used
    files = glob(os.path.join(DEPLOYDIR, case, 'models', run, '*.pkl'))
    skip  = int(cadence / Resolution[case])
    files = [files[i] for i in range(0, forecasts+1, skip)]

    logger.debug('Models to be used:')
    for i, file in enumerate(files):
        logger.debug(f'- {i:02d}: {file}')

    logger.info('Loading features for this set')
    features = utils.load_pkl(os.path.join(DEPLOYDIR, case, 'features', f'{run}.pkl'))

    logger.info(f'Forecasting for run {run}')
    fcs = pd.DataFrame(index=df.index)
    for i in tqdm(range(forecasts+1), desc='Forecasting'):
        fc      = i*cadence
        model   = utils.load_pkl(files[i])
        fcs[fc] = model.predict(df[features[fc]])
        del model

    logger.info('Saving forecasts')
    output = os.path.join(DEPLOYDIR, case, 'forecasts', f'{run}.csv')
    flags  = {}
    if glob(output):
        flags = {
            'mode'  : 'a',
            'header': False
        }

    fcs.to_csv(output, **flags)

def main(case, forecasts, cadence, input, key, nonoptimize, skip_check):
    """
    Main function that handles data loading, stepping through data processing
    functions, and data masking for forecasting.

    Parameters
    ----------
    case: string
        The case being processed. Currently supports: r0, pwv
    forecasts: int
        How far into the future to forecast
    cadence: int
        How often to forecast
    input: string
        Path to the input data h5
    key: string
        String key to the data in the input h5
    nonoptimize: bool
        Disables mask optimization which only applies the best models for each
        timestamp
    skip_check: bool
        Skips the check for a last_window.pkl, effectively forcing a full
        reprocess of any input data

    Returns
    -------
    bool
        Status of the script. True for success, False for failure.
    """
    # Load the data in
    try:
        if input.endswith('.csv'):
            df = pd.read_csv(input)
            df.index = pd.DatetimeIndex(df.datetime)
            df.drop(columns=['datetime'])
        elif input.endswith('.h5'):
            df = pd.read_hdf(input, key)
    except:
        logger.exception(f'Unable to load the input data from {input} using key {key}')
        return False

    if not skip_check:
        # Only process timestamps that haven't been before
        try:
            logger.info('Remove already processed data')
            last       = utils.load_pkl(os.path.join(DEPLOYDIR, case, '_data', 'last_window.pkl'))
            index      = df.index == last
            index[-9:] = True
            df         = df[index]
        except:
            logger.exception('Unable to load last window index, processing all possible windows')

    # Insert calculated features
    df = calculate_features(df)

    # Extract features from windows
    df = extract_features(df, case)

    # If no extractions happened, return
    if df is False:
        return False

    # Create masks determining which indices go to which set of models
    logger.debug('Creating run masks')
    masks = {}
    if case == 'r0':
        # Find indices where r0 and/or Cn2 were not-nan
        isvalid = ~df.drop(columns=set(df.columns) - set(['historical_feature_r0_10T', 'log_Cn2_10T']), errors='ignore').isna()

        # Check if r0 and Cn2 were present
        r0  = 'historical_feature_r0_10T' in isvalid
        Cn2 = 'log_Cn2_10T' in isvalid

        # Optimize only selects the best model for each window
        if not nonoptimize:
            # r0 and Cn2 are present
            if r0 and Cn2:
                masks['r0.Cn2.weather.historical'] = df[ isvalid['historical_feature_r0_10T'] &  isvalid['log_Cn2_10T']] #  r0 &  Cn2
                masks['r0.weather.historical']     = df[ isvalid['historical_feature_r0_10T'] & ~isvalid['log_Cn2_10T']] #  r0 & !Cn2
                masks['r0.Cn2.weather']            = df[~isvalid['historical_feature_r0_10T'] &  isvalid['log_Cn2_10T']] # !r0 &  Cn2
                masks['r0.weather']                = df[~isvalid['historical_feature_r0_10T'] & ~isvalid['log_Cn2_10T']] # !r0 & !Cn2
            # r0 is present but Cn2 is not
            elif r0 and not Cn2:
                masks['r0.weather.historical']     = df[ isvalid['historical_feature_r0_10T']] # r0 is present
                masks['r0.weather']                = df[~isvalid['historical_feature_r0_10T']] # r0 is not present
            # r0 is not present but Cn2 is
            elif not r0 and Cn2:
                masks['r0.Cn2.weather']            = df[ isvalid['log_Cn2_10T']] # Cn2 is present
                masks['r0.weather']                = df[~isvalid['log_Cn2_10T']] # Cn2 is not present
            # Neither r0 nor Cn2 is present
            else:
                masks['r0.weather']                = df

        # A specific run has been chosen
        elif isinstance(nonoptimize, str):
            if nonoptimize == 'r0.Cn2.weather.historical':
                if r0 and Cn2:
                    masks['r0.Cn2.weather.historical'] = df[isvalid['historical_feature_r0_10T'] & isvalid['log_Cn2_10T']]
            elif nonoptimize == 'r0.weather.historical'
                if r0:
                    masks['r0.weather.historical'] = df[isvalid['historical_feature_r0_10T']]
            elif nonoptimize == 'r0.Cn2.weather':
                if Cn2:
                    masks['r0.Cn2.weather'] = df[isvalid['log_Cn2_10T']]
            elif nonoptimize == 'r0.weather':
                masks['r0.weather'] = df
            else:
                logger.error(f'Run {nonoptimize} does not exist for this case ({case}), should be one of: [r0.Cn2.weather.historical, r0.weather.historical, r0.Cn2.weather, r0.weather]')

        # Unoptimized selects all windows that each model type could process
        else:
            # r0 and Cn2 are present
            if r0 and Cn2:
                masks['r0.Cn2.weather.historical'] = df[isvalid['historical_feature_r0_10T'] & isvalid['log_Cn2_10T']]
            if r0:
                masks['r0.weather.historical'] = df[isvalid['historical_feature_r0_10T']]
            if Cn2:
                masks['r0.Cn2.weather'] = df[isvalid['log_Cn2_10T']]
            masks['r0.weather'] = df
    elif case == 'pwv':
        # PWV doesn't have multiple sets of models
        masks['full'] = df
    elif case in ['temperature', 'pressure', 'relative_humidity', 'wind_speed']:
        # Weather cases also only have one set of models
        masks['full'] = df

    # Predict
    for run, data in masks.items():
        logger.info(f'Performing forecasting using model set {run}')
        logger.debug(f'{run} contains {data.index.size / df.index.size * 100:.2f}% of the windows')
        forecast(case, run, data, forecasts, cadence)

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('case',                 type     = str,
                                                choices  = ['r0', 'pwv', 'temperature', 'pressure','relative_humidity', 'wind_speed'],
                                                help     = 'The kind of run this is'
    )
    parser.add_argument('-f', '--forecasts',    type     = int,
                                                help     = '''\
How far to forecast. This value should be N * cadence <= [r0,weather=180|pwv=300]. Examples:
- r0 : With cadence =  5, to forecast 3 hours do 180/5  = 36
- r0 : With cadence = 10, to forecast 3 hours do 180/10 = 18
- pwv: With cadence = 30, to forecast 3 hours do 180/30 = 6
- pwv: With cadence = 60, to forecast 3 hours do 180/60 = 2
Defaults to 3 hours for any cadence in any case.\
'''
    )
    parser.add_argument('-c', '--cadence',      type     = int,
                                                help     = 'The forecasting cadence. r0 must be a multiple of 5, pwv must be a multiple of 30. Defaults to 5 for r0 and 30 for pwv.'
    )
    parser.add_argument('-i', '--input',        type     = str,
                                                # required = True, TODO: Revert and remove default
                                                default  = 'deployment/test.h5',
                                                help     = 'Path to the input data file'
    )
    parser.add_argument('-d', '--deploydir',    type     = str,
                                                default  = os.getenv('MLOC_DEPLOYDIR'),
                                                help     = 'Sets the deployment directory for this run. If not set, defaults to the environment variable MLOC_DEPLOYDIR'
    )
    parser.add_argument('-k', '--key',          type     = str,
                                                default  = 'test',
                                                help     = 'Key to the Pandas DataFrame object in the --input file if it is an H5'
    )
    parser.add_argument('-no', '--nonoptimize', type     = str,
                                                nargs    = '?',
                                                const    = True,
                                                default  = False,
                                                help     = '''\
Disables optimization of forecasting which selects the best model for a given forecast if multiple cases are available. If disabled, \
all models will be applied to all forecasts, if viable. If this option is followed by a string with the same name as a run for this case, \
only the models for that run will be used.
'''
    )
    parser.add_argument('-p', '--preview',      action   = 'store_true',
                                                help     = 'Previews the arguments for the user'
    )
    parser.add_argument('--skip_check',         action   = 'store_true',
                                                help     = 'Skips the check for the last window processed. May reprocess already processed windows'
    )
    parser.add_argument('--debug',              action   = 'store_true',
                                                help     = 'Enables debug logging'
    )
    parser.add_argument('--logfile',            type     = str,
                                                help     = 'Enables logging to a file'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level   = logging.DEBUG if (args.debug or args.preview) else logging.INFO,
        format  = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt = '%m-%d %H:%M',
        stream  = sys.stdout
    )
    logger = logging.getLogger('mloc/forecast.py')

    if args.logfile:
        filehl = logging.FileHandler(args.logfile)
        filehl.setLevel(logging.DEBUG)
        logger.addHandler(filehl)

    # Verify the cadence
    default_cadence = Resolution[args.case]
    if args.cadence is not None:
        if args.cadence % default_cadence != 0:
            logger.error(f'The --cadence (-c) for {args.case} must be a multiple of {default_cadence}. You gave {args.cadence}')
            sys.exit(0)
    else:
        args.cadence = default_cadence

    # Verify the forecast
    default_fcs = int(180 / args.cadence)
    if args.forecasts is not None:
        fc = args.forecasts * args.cadence
        if fc > 180:
            logger.warning(f'The --forecasts (-f) exceeds 180 minutes for the given cadence ({args.forecasts} * {args.cadence} = {fc}). Lowering forecasts to 180/{args.cadence} = {default_fcs}')
            args.forecasts = default_fcs
    else:
        args.forecasts = default_fcs

    # Verify the deployment directory
    if args.deploydir:
        # Verify the case directory exists
        if args.case in glob(os.path.join(args.deploydir, '*')):
            # Check which directories are available
            folders = glob(os.path.join(args.deploydir, args.case, '*'))
            for folder in ['_data', 'features', 'models', 'forecasts']:
                if folder not in folders:
                    logger.warning(f'Directory {folder} not found in {args.deploydir}/{args.case}/')

                    # Some folders can be generated
                    if folder in ['_data', 'forecasts']:
                        os.mkdir(os.path.join(args.deploydir, args.case, folder))
                        logger.info(f'Created directory {args.deploydir}/{args.case}/{folder}')

                    # features/ and models/ needs to be populated by the user
                    else:
                        logger.error(f'Cannot create directory {folder}, it must be manually created and contain the expected contents')
                        sys.exit(0)
        else:
            logger.error(f'Directory for case {args.case} not found in the deployment directory ({args.deploydir}), please create it and add the features and models subdirectories')
            sys.exit(0)
    else:
        logger.error(f'No deployment directory found. Please set it either via the environment variable MLOC_DEPLOYDIR or via -d, --deploydir')
        sys.exit(0)

    # Set the global
    DEPLOYDIR = args.deploydir

    # Log the arguments
    logger.debug(f'Forecasting case  : {args.case}')
    logger.debug(f'Using forecasts   : {args.forecasts}')
    logger.debug(f'Using cadence     : {args.cadence}')
    logger.info(f'Forecasting {args.case} every {args.cadence} minutes up to {args.forecasts * args.cadence} minutes')
    logger.debug(f'Input file        : {args.input}')
    logger.debug(f'Input key         : {args.key}')
    logger.debug(f'Optimized         : {args.nonoptimize}')
    logger.debug(f'Skip Check        : {args.skip_check}')

    # If previewing the arguments, exit
    if args.preview:
        sys.exit(1)

    try:
        status = main(args.case, args.forecasts, args.cadence, args.input, args.key, args.nonoptimize, args.skip_check)

        if status:
            logger.info('Finished successfully')
        else:
            logger.error('Failed to complete')
    except Exception:
        logger.exception('Failed to complete')
