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

# External
from mlky import Config as C


Logger = logging.getLogger('timefed/pipeline')


def main():
    """
    Executes the TimeFED pipeline
    """
    Logger.info('Executing pipeline for scripts:')
    for key in ['preprocess', 'extract', 'subselect', 'model']:
        Logger.info(f'- {key} is {"not " if not C[key].enabled else ""}enabled')

    if C.preprocess.enabled:
        if C.preprocess.kind == 'dsn':
            from timefed.preprocess.dsn import main as preprocess
        elif C.preprocess.kind == 'mloc':
            from timefed.preprocess.mloc import main as preprocess

        Logger.info(f'Running script: preprocess[{C.preprocess.kind}]')
        preprocess()
    else:
        Logger.debug('The preprocess script is disabled')

    if C.extract.enabled:
        from timefed.core.extract import main as extract

        Logger.info(f'Running script: extract')
        extract()
    else:
        Logger.debug('The extract script is disabled')

    if C.subselect.enabled:
        from timefed.core.subselect import main as subselect

        Logger.info(f'Running script: subselect')
        subselect()
    else:
        Logger.debug('The subselect script is disabled')

    if C.model.enabled:
        if C.model.algorithm == 'RandomForest':
            from timefed.core.rf import main as model
        elif C.model.algorithm == 'LSTM':
            from timefed.core.lstm import main as model

        Logger.info(f'Running script: model')
        model()
    else:
        Logger.debug('The model script is disabled')

    return True


if __name__ == '__main__':
    Logger.error('Calling this script directly has been deprecated, please use the timefed CLI')
