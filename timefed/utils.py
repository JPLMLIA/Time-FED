"""
"""
import logging
import multiprocessing as mp
import multiprocessing.pool as mp_pool
import pickle
import sys

from datetime import datetime as dtt

from timefed.config import Config

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
    config = Config(args.config, args.section)

    levels = {
        'critical': logging.CRITICAL,
        'error'   : logging.ERROR,
        'warning' : logging.WARNING,
        'info'    : logging.INFO,
        'debug'   : logging.DEBUG
    }

    logging.basicConfig(
        level   = levels.get(config.log.level or '', logging.DEBUG),
        format  = config.log.format  or '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt = config.log.datefmt or '%m-%d %H:%M',
        stream  = sys.stdout,
        force   = True
    )

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

class NoDaemonProcess(mp.Process):
    '''
    Credit goes to Massimiliano
    https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
    '''
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonContext(type(mp.get_context())):
    '''
    Credit goes to Massimiliano
    https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
    '''
    Process = NoDaemonProcess

class Pool(mp_pool.Pool):
    '''
    Credit goes to Massimiliano
    https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
    '''
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super().__init__(*args, **kwargs)
