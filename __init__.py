import logging
import sys

logging.basicConfig(
    level   = logging.DEBUG,
    format  = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt = '%m-%d %H:%M',
    stream  = sys.stdout
)
