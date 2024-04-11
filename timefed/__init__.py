import click

from mlky import Config

# Disable converting lists to dicts and some mlky warnings
Config._opts.convertListTypes = False
Config.Null._warn = False


def config_options(func):
    @click.option('-c', '--config',
        help     = 'Configuration YAML',
        required = True
    )
    @click.option('-p', '--patch',
        help    = 'Sections to patch with',
        default = 'generated'
    )
    @click.option('-d', '--defs',
        help    = 'Definitions file',
        default = 'configs/defs.yml'
    )
    @click.option('--print',
        help    = 'Prints the Configuration to terminal',
        is_flag = True
    )
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
