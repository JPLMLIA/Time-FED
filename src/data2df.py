"""
Usage: data2df.py
Required arguments:

"""

import argparse
import logging
import sys

from utils import load_weather, load_bls, load_r0

#
# CONSTANTS
#

VALID_DATASETS = ['weather', 'cn2', 'r0-day', 'r0-night']

#
# SUBROUTINES
#


# MAIN
#

def main(input, output, dataset):

    # validate command line options
    #
    if dataset not in VALID_DATASETS:
        logger.error("Received invalid input for dataset: {}".format(dataset))
        return False

    if dataset == 'weather':
        df = load_weather(input)
    elif dataset == 'cn2':
        df = load_bls(input)
    elif dataset == 'r0-day':
        df = load_r0(input, 'day')
    else:
        df = load_r0(input, 'night')

    df.to_hdf(output, dataset)

    return True


if __name__ == '__main__':

    # initalize argparse
    #
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-i', '--input', required=True, help='Path to data')
    parser.add_argument('-o', '--output', required=True, help='Path to .h5 output file')
    parser.add_argument('-d', '--dataset', required=True, help=str(VALID_DATASETS))


    args = parser.parse_args()
    # initialize logger
    #
    logger = logging.getLogger('data2df')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        stream=sys.stdout)  # ,
    if main(**vars(args)):
        logger.info("Finished successfully")
        sys.exit(0)
    else:
        logger.error("Error detected.  Aborting program.")
        sys.exit(-1)


