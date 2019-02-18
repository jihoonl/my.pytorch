
import logging

#formatter = logging.Formatter(
#    '[%(levelname)s] %(asctime)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
formatter = logging.Formatter(
    '[%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
ch = logging.StreamHandler()
ch.setFormatter(formatter)

logger = logging.getLogger('capsnet')
logger.setLevel(logging.INFO)
logger.addHandler(ch)

def set_debug(debug=True):
    global logger

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
