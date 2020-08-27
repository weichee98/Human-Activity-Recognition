import logging


class Log:
    @staticmethod
    def setup_logger(name, log_file, level=logging.INFO, mode='w'):
        handler = logging.FileHandler(log_file, mode=mode)
        handler.setFormatter(logging.Formatter('%(asctime)-8s %(message)s'))
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        return logger
