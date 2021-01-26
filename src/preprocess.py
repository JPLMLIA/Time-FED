import argparse
import logging
import pandas as pd
import numpy  as np
import os

# Import utils to set the logger
import utils

def subselect(a, b, df):
    """
    Subselects from a dataframe between dates

    Parameters
    ----------
    a : str
        The first date
    b : str
        Either the second date or a conditional for the first date
    df : pandas.DataFrame
        The dataframe to subselect from

    Returns
    -------
    sub : pandas.DataFrame
        The subselected dataframe
    """
    if b in ['<', '<=', '>', '>=']:
        if b == '<':
            sub = df[df.index < a]
        elif b == '<=':
            sub = df[df.index <= a]
        elif b == '>':
            sub = df[df.index > a]
        elif b == '>=':
            sub = df[df.index >= a]
        else:
            logger.error(f'Unsupported comparison type: {b}')
    else:
        sub = df[(a <= df.index) & (df.index < b)]

    return sub

def preprocess(df, output=None, key_out=None, between=None, train=None, test=None, **kwargs):
    """
    Preprocesses the dataframe with additional features

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to apply new features to
    output : str
        Path to output.h5; Optional, will only write if provided
    key_out : str
        The key to use for the dataframe when writing to the output.h5
    between : tuple of str
        Subselects between two dates: DATE1 <= time < DATE2
    train : tuple of str
        Subselects the processed dataframe as a training dataframe. Requires 2 strings, the first always being a date and the second being either a date or conditional. Usage examples:
         Select after a date: (date, <)
         Select before a date: (date, >)
         Select between [date1, date2): (date1, date2)
    test : tuple of str
        Subselects the processed dataframe as a testing dataframe. Requires 2 strings, the first always being a date and the second being either a date or conditional. Usage examples:
         Select after a date: (date, <)
         Select before a date: (date, >)
         Select between [date1, date2): (date1, date2)

    Returns
    -------
    """
    if between:
        logger.info(f'Subselecting between [{between[0]}, {between[1]})')
        df = subselect(*between, df).copy()

    logger.debug(f'df.describe():\n{df.describe()}')

    logger.info('Creating new features')
    df['month']  = df.index.month
    df['day']    = df.index.dayofyear
    df['logCn2'] = np.log10(df['Cn2'])

    logger.debug(f'Count of non-NaN values:\n{(~df.isnull()).sum()}')

    if output:
        df.to_hdf(output, key_out)

    if train:
        logger.info(f'Creating training subset using {train}')
        train = subselect(*train, df)
        logger.debug(f'Count of non-NaN values for train:\n{(~train.isnull()).sum()}')
        if output:
            train.to_hdf(output, 'train')

    if test:
        logger.info(f'Creating testing subset using {test}')
        test = subselect(*test, df)
        logger.debug(f'Count of non-NaN values for test:\n{(~test.isnull()).sum()}')
        if output:
            test.to_hdf(output, 'test')

    return df, train, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--input',    type     = str,
                                            required = True,
                                            metavar  = '/path/to/data.h5',
                                            help     = 'Path to input.h5'
    )
    parser.add_argument('-ki', '--key_in',  type     = str,
                                            metavar  = 'KEY',
                                            help     = 'The key of the dataframe in the input.h5'
    )
    parser.add_argument('-o', '--output',   type     = str,
                                            metavar  = '/path/to/output.h5',
                                            help     = 'Path to output.h5; Optional, will only write if provided'
    )
    parser.add_argument('-ko', '--key_out', type     = str,
                                            default  = 'preprocess',
                                            metavar  = 'KEY',
                                            help     = 'The key to use for the dataframe when writing to the output.h5'
    )
    parser.add_argument('-b', '--between',  type     = str,
                                            nargs    = 2,
                                            metavar  = ('DATE1', 'DATE2'),
                                            help     = 'Subselects between two dates: DATE1 <= time < DATE2'
    )
    parser.add_argument('--train',          type     = str,
                                            nargs    = 2,
                                            metavar  = ('DATE', 'ARG'),
                                            help     = 'Subselects the processed dataframe as a training dataframe. Requires 2 strings, the first always being a date and the second being either a date or conditional. Usage examples:' \
                                                     + '\n\tSelect after a date: --train date <' \
                                                     + '\n\tSelect before a date: --train date >' \
                                                     + '\n\tSelect between [date1, date2): --train date1 date2'
    )
    parser.add_argument('--test',           type     = str,
                                            nargs    = 2,
                                            metavar  = ('DATE', 'ARG'),
                                            help     = 'Subselects the processed dataframe as a testing dataframe. Requires 2 strings, the first always being a date and the second being either a date or conditional. Usage examples:' \
                                                     + '\n\tSelect after a date: --test date <' \
                                                     + '\n\tSelect before a date: --test date >' \
                                                     + '\n\tSelect between [date1, date2): --train date1 date2'
    )


    args = parser.parse_args()

    logger = logging.getLogger(os.path.basename(__file__))

    try:
        df = pd.read_hdf(args.input, args.key_in)

        preprocess(df, **vars(args))

        logger.info('Finished successfully')
    except Exception as e:
        logger.exception('Failed to complete')
