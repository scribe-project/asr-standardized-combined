import logging
import os

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        '%(levelname)s: %(name)s - %(message)s'
    ))
    logger.addHandler(ch)

def create_new_logger(new_logger, name):
    name = filename_to_loggingname(name)
    new_logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        '%(levelname)s: {} - %(message)s'.format(name)
    ))
    new_logger.addHandler(ch)
    return new_logger

def filename_to_loggingname(filename):
    # assumption is that asr_standardized_combined occurs 2x in path e.g. .../asr_standardized_combined/asr_standardized_combined/standardize/standardize_nbtale12.py
    package_name = 'asr_standardized_combined'
    # get everything after the first package name
    filename = os.path.normpath(filename)
    return '.'.join([part for part in filename[filename.index(package_name) + len(package_name):].strip('.py').split(os.sep) if part])