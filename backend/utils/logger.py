
import logging
import sys

def setup_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger # Avoid adding multiple handlers

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
