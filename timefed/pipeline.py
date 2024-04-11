import logging
import os
import sys

import click

from mlky import Config

from timefed import config_options


Logger = logging.getLogger('timefed/pipeline.py')


def main():
    """\
    Executes the TimeFED pipeline
    """
    Logger.info('Executing pipeline for scripts:')
    for key in ['preprocess', 'extract', 'subselect', 'model']:
        Logger.info(f'  {key} is {"not " if not Config[key].enabled else ""}enabled')

    if Config.preprocess.enabled:
        if Config.preprocess.kind == 'dsn':
            from timefed.preprocess.dsn import main as preprocess
        elif Config.preprocess.kind == 'mloc':
            from timefed.preprocess.mloc import main as preprocess

        Logger.info(f'Running script: preprocess[{Config.preprocess.kind}]')
        preprocess()
    else:
        Logger.debug('The preprocess script is disabled')

    if Config.extract.enabled:
        from timefed.core.extract import main as extract

        Logger.info(f'Running script: extract')
        extract()
    else:
        Logger.debug('The extract script is disabled')

    if Config.subselect.enabled:
        from timefed.core.subselect import main as subselect

        Logger.info(f'Running script: subselect')
        subselect()
    else:
        Logger.debug('The subselect script is disabled')

    if Config.model.enabled:
        from timefed.core.model import main as model

        Logger.info(f'Running script: model')
        model()
    else:
        Logger.debug('The model script is disabled')

    return True


@click.command(name='run')
@config_options
def run(config, patch, defs, print):
    """\
    Executes the TimeFED pipeline
    """
    # Initialize the global Configuration object
    Config(config, patch, defs=defs)

    # Print configuration to terminal
    if print:
        click.echo(Config.dumpYaml())

    # Logging handlers
    handlers = []

    # Create console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(getattr(logging, Config.log.terminal))
    handlers.append(sh)

    if Config.log.file:
        if Config.log.mode == 'write' and os.path.exists(Config.log.file):
            os.remove(Config.log.file)

        # Add the file logging
        fh = logging.FileHandler(Config.log.file)
        fh.setLevel(Config.log.level)
        handlers.append(fh)

    logging.basicConfig(
        level    = getattr(logging, Config.log.level),
        format   = Config.log.format  or '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt  = Config.log.datefmt or '%m-%d %H:%M',
        handlers = handlers
    )

    if Config.validate():
        main()
    else:
        Logger.error('Please correct any Configuration errors before proceeding')


if __name__ == '__main__':
    run()
