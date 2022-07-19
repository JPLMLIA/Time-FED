"""
"""
import argparse
import logging

from timefed.config import Config
from timefed.utils  import init

def main(section):
    """
    """
    config = Config()

    if config.preprocess.run:
        Logger.info(f'Running preprocess using section key {config.preprocess.key}')
        from timefed.research.dsn.preprocess import main as preprocess
        config.reset(config.preprocess.key)
        preprocess()
        config.reset(section)

    if config.extract.run:
        Logger.info(f'Running extract using section key {config.extract.key}')
        from timefed.research.extract import process as extract
        config.reset(config.extract.key)
        extract()
        config.reset(section)

    if config.select.run:
        Logger.info(f'Running select using section key {config.select.key}')
        from timefed.research.select_features import select
        config.reset(config.select.key)
        select()
        config.reset(section)

    if config.classify.run:
        Logger.info(f'Running classify using section key {config.classify.key}')
        from timefed.research.classify import classify
        config.reset(config.classify.key)
        classify()

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/config.yaml',
                                            help     = 'Path to a config.yaml file'
    )
    parser.add_argument('-s', '--section',  type     = str,
                                            default  = 'pipeline',
                                            metavar  = '[section]',
                                            help     = 'Section of the config to use'
    )

    args  = parser.parse_args()
    state = False

    # Initialize the loggers
    init(args)
    Logger = logging.getLogger('timefed/pipeline.py')

    # Process
    try:
        state = main(args.section)
    except Exception:
        Logger.exception('Caught an exception during runtime')
    finally:
        if state is True:
            Logger.info('Finished successfully')
        else:
            Logger.info(f'Failed to complete with status code: {state}')
