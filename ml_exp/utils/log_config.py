import logging
import datetime
import pytz
from functools import wraps


class CustomFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created, tz=pytz.timezone("America/Sao_Paulo"))

        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.isoformat()


class AppLogger(logging.Logger):
    """Custom logger class for service layer operations within the context of the API.

    Arguments:
    name (str): The name of the logger.
    level (Union[int, str], optional): The logging level. Defaults to logging.NOTSET.
    """

    def __init__(self, name: str, level: int | str = logging.NOTSET) -> None:
        super().__init__(name, level)

        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = CustomFormatter(fmt="%(asctime)s : %(levelname)s : %(message)s",
                                    datefmt="%Y-%m-%d %H:%M:%S")
        
        handler.setFormatter(formatter)
        self.addHandler(handler)


class LogService:
    __loggers = {}

    def get_logger(self, name: str):
        """Retrieves or creates a logger for the application with a specific name

        Args:
        name (str): The name of the logger to be retrieved/created.

        Returns:
        AppLogger (logging.Logger): The logger instance associated with the given name.
        """

        if name not in self.__loggers:
            logger = AppLogger(name)
            logger.setLevel(logging.INFO)
            self.__loggers[name] = logger
        return self.__loggers[name]


def handle_exceptions(logger):
    """Decorator to log exceptions and capture unexpected errors that affect the execution and completion of the operation in the service layer."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                logger.debug(f"Starting function execution {func.__name__}")
                response = func(*args, **kwargs)
                logger.debug(f"Successful execution of the function {func.__name__}")
                return response
            except Exception as error:
                exception_type = error.__class__.__name__
                logger.critical("Unexpected Critical Error at %s: %s",
                                func.__name__,
                                exception_type,
                                exc_info=error)
                return ("UnexpectedError", f"{exception_type}: {error}")
        return wrapper
    return decorator