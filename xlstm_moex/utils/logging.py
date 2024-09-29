import logging


def init_logger(name: str = __name__, level: str = 'INFO') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level=level)
    
    handler = logging.StreamHandler()
    handler.setLevel(level=level)

    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
