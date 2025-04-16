# Copyright 2025, by the California Institute of Technology. ALL RIGHTS RESERVED.
# United States Government Sponsorship acknowledged. Any commercial use must be
# negotiated with the Office of Technology Transfer at the California Institute of
# Technology.
#
# This software may be subject to U.S. export control laws. By accepting this software,
# the user agrees to comply with all applicable U.S. export laws and regulations. User
# has the responsibility to obtain export licenses, or other export authority as may be
# required before exporting such information to foreign countries or providing access
# to foreign persons.

# Builtin
import logging
import os
import sys

from pathlib import Path

# External
import click
import mlky

from mlky import Config as C


Logger = logging.getLogger('timefed/cli')


def initConfig(config, patch, defs, override, printconfig=False, printonly=False, print=print):
    """
    Initializes the mlky Config object
    """
    C(config, _patch=patch, _defs=defs, _override=override)

    # Print configuration to terminal
    if printconfig or printonly:
        print(f'Config({config!r}, _patch={patch!r}, _defs={defs})')
        print('-'*100)
        print(C.toYaml(comments=None, listStyle='short', header=False))
        print('-'*100)

        if printonly:
            sys.exit()


def initLogging():
    """
    Initializes the logging module per the config
    """
    # Logging handlers
    handlers = []

    # Create console handler
    sh = logging.StreamHandler(sys.stdout)

    if (level := C.log.terminal):
        sh.setLevel(level)

    handlers.append(sh)

    if (file := C.log.file):
        if C.log.mode == 'write' and os.path.exists(file):
            os.remove(C.log.file)

        # Make sure path exists
        Path(file).parent.mkdir(exist_ok=True, parents=True)

        # Add the file logging
        fh = logging.FileHandler(file)
        fh.setLevel(C.log.level or logging.DEBUG)

        handlers.append(fh)

    logging.basicConfig(
        level    = C.log.get('level', 'DEBUG'),
        format   = C.log.get('format', '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'),
        datefmt  = C.log.get('format', '%m-%d %H:%M'),
        handlers = handlers,
    )


@click.group(name='timefed')
def cli():
    """\
    TimeFED is a machine learning system for Time series Forecasting, Evaluation and Deployment.
    """
    ...


# Path to the mlky definitions file for timefed
defs = Path(__file__).parent / 'configs/defs/defs.yml'

@cli.command(name='run', context_settings={'show_default': True})
@click.option('-s', '--script', type=click.Choice(['preprocess', 'extract', 'subselect', 'model']), help='Executes a single script. If not provided, entire pipeline is executed')
@mlky.cli.config
@mlky.cli.patch
@mlky.cli.defs(default=defs)
@mlky.cli.override
@click.option('-dv', '--disableValidate', help='Disables the validation requirement. Validation will still be occur, but execution will not be prevented', is_flag=True)
@click.option("-pc", "--printConfig", help="Prints the configuration to terminal and continues", is_flag=True)
@click.option("-po", "--printOnly", help="Prints the configuration to terminal and exits", is_flag=True)
def main(script, disablevalidate, **kwargs):
    """\
    Executes the TimeFED pipeline
    """
    initConfig(**kwargs, print=click.echo)
    initLogging()

    if C.validateObj() or disablevalidate:
        match script:
            case None:
                from timefed.pipeline import main

            case 'preprocess':
                match (kind := C.preprocess.kind):
                    case 'dsn':
                        from timefed.preprocess.dsn import main
                        Logger.info('Executing only script: preprocess[dsn]')
                    case 'mloc':
                        from timefed.preprocess.mloc import main
                        Logger.info('Executing only script: preprocess[mloc]')
                    case _:
                        raise AttributeError(f'Unknown preprocess.kind: {kind}')

            case 'extract':
                from timefed.core.extract import main
                Logger.info('Executing only script: extract')

            case 'subselect':
                from timefed.core.subselect import main
                Logger.info('Executing only script: subselect')

            case 'model':
                from timefed.core.model import main
                Logger.info('Executing only script: model')

            case _:
                raise AttributeError(f'Unknown script choice: {script}')

        main()
    else:
        Logger.error('Please correct the configuration errors before proceeding')


# Add mlky as subcommands
mlky.cli.setDefaults(patch='generated', defs=defs)
cli.add_command(mlky.cli.commands)


if __name__ == '__main__':
    main()
