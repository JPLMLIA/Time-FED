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
        from timefed.core.model import main as model

        Logger.info(f'Running script: model')
        model()
    else:
        Logger.debug('The model script is disabled')

    return True


if __name__ == '__main__':
    Logger.error('Calling this script directly has been deprecated, please use the timefed CLI')
