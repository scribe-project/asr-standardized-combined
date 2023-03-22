import logging

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter(
    '%(levelname)s: %(name)s - %(message)s'
))
logger.addHandler(ch)