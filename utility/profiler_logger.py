import os
import logging


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    def __init__(self):
        self._logger = logging.getLogger("profiler")
        self._logger.setLevel(logging.DEBUG)

        if not os.path.isdir("./.log"):
            os.mkdir("./.log")

        ch = logging.FileHandler("./.log/profiler.log", mode="a")
        ch.setLevel(logging.DEBUG)

        ch.setFormatter(CustomFormatter())

        self._logger.addHandler(ch)

    def get_logger(self):
        return self._logger


LOGGER = Logger().get_logger()
