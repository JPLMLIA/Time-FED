"""
"""
import logging
import os
import pickle
import sys
from pathlib import Path

from datetime import datetime as dtt
from mlky     import Config

# Increase matplotlib's logger to warning to disable the debug spam it makes
logging.getLogger('matplotlib').setLevel(logging.WARNING)

Logger = logging.getLogger('timefed/utils.py')

def init(args):
    """
    Initializes the root logger with parameters defined by the config.

    Parameters
    ----------
    config: timefed.config.Config
        MilkyLib configuration object

    Notes
    -----
    config keys:
        log:
            level: str
            format: str
            datefmt: str
    """
    config = Config(args.config, _patch=args.section)

    levels = {
        'critical': logging.CRITICAL,
        'error'   : logging.ERROR,
        'warning' : logging.WARNING,
        'info'    : logging.INFO,
        'debug'   : logging.DEBUG
    }

    handlers = []

    # Create console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(levels.get(config.log.level or '', logging.INFO))
    handlers.append(sh)

    if config.log.file:
        if config.log.reset and os.path.exists(config.log.file):
            os.remove(config.log.file)

        # Make sure path exists
        Path(config.log.file).parent.mkdir(exist_ok=True, parents=True)

        # Add the file logging
        fh = logging.FileHandler(config.log.file)
        fh.setLevel(logging.DEBUG)
        handlers.append(fh)

    logging.basicConfig(
        level    = logging.DEBUG,
        format   = config.log.format  or '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt  = config.log.datefmt or '%m-%d %H:%M',
        handlers = handlers,
        force    = True
    )

    logging.getLogger().debug(f'Logging initialized using Config({args.config}, {args.section})')

    if config.log.config:
        shutil.copy(config._flags.file, config.log.config)

def timeit(func):
    """
    Utility decorator to track the processing time of a function

    Parameters
    ----------
    func : function
        The function to track

    Returns
    -------
    any
        Returns the return of the tracked function
    """
    def _wrap(*args, **kwargs):
        start = dtt.now()
        ret   = func(*args, **kwargs)
        Logger.debug(f'Finished function {func.__name__} in {(dtt.now() - start).total_seconds()} seconds')
        return ret
    # Need to pass the docs on for sphinx to generate properly
    _wrap.__doc__ = func.__doc__
    return _wrap

def save_pkl(file, data):
    """
    Saves data to a file via pickle

    Parameters
    ----------
    file : str
        Path to a file to dump the data to via pickle
    data: any
        Any pickleable object
    """
    with open(file, 'wb') as file:
        pickle.dump(data, file)

def load_pkl(file):
    """
    Loads data from a pickle

    Parameters
    ----------
    file : str
        Path to a Python pickle file to load

    Returns
    -------
    any
        The data object loaded from the pickle file
    """
    return pickle.load(open(file, 'rb'))

def align_print(iterable, enum=False, delimiter='=', offset=1, prepend='', print=print):
    """
    Pretty prints an iterable in the form {key} = {value} such that the delimiter (=)
    aligns on each line

    Parameters
    ----------
    iterable: iterable
        Any iterable with a .items() function
    enum: bool, default = False
        Whether to include enumeration of the items
    delimiter, default = '='
        The symbol to use between the key and the value
    offset: int, default = 1
        Space between the key and the delimiter: {key}{offset}{delimiter}
        Defaults to 1, eg: "key ="
    prepend: str, default = ''
        Any string to prepend to each line
    print: func, default = print
        The print function to use. Allows using custom function instead of Python's normal print
    """
    # Determine how much padding between the key and delimiter
    pad = max([1, len(max(iterable.keys(), key=len))]) + offset

    # Build the formatted string
    fmt = prepend
    if enum:
        fmt += '- {i:' + f'{len(str(len(iterable)))}' + '}: '
    fmt += '{key:'+ str(pad) + '}' + delimiter + ' {value}'

    # Create the formatted list
    fmt_list = []
    for i, (key, value) in enumerate(iterable.items()):
        string = fmt.format(i=i, key=key, value=value)
        fmt_list.append(string)

    for string in fmt_list:
        print(string)

    return fmt_list
