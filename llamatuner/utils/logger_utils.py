import logging
import os
import sys
from logging import Formatter, LogRecord
from pathlib import Path
from typing import Optional, Union

import torch.distributed as dist
from colorama import Fore, Style

logger_initialized: dict = {}


class ColorfulFormatter(Formatter):
    """Formatter that adds ANSI color codes to log messages based on their
    level.

    Attributes:
        COLORS: Dictionary mapping log levels to their corresponding color codes

    Example:
        >>> formatter = ColorfulFormatter('%(levelname)s: %(message)s')
        >>> handler = logging.StreamHandler()
        >>> handler.setFormatter(formatter)
    """

    COLORS = {
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
        'DEBUG': Fore.LIGHTGREEN_EX,
    }

    def format(self, record: LogRecord) -> str:
        record.rank = int(os.getenv('LOCAL_RANK', '0'))
        log_message = super().format(record)
        return self.COLORS.get(record.levelname, '') + log_message + Fore.RESET


def get_logger(
    name: str,
    log_file: Optional[Union[str, Path]] = None,
    log_level: int = logging.INFO,
    file_mode: str = 'w',
) -> logging.Logger:
    """Initialize and get a logger by name with optional file output.

    This function creates or retrieves a logger with the specified configuration.
    It handles distributed training scenarios by managing log levels across different
    process ranks and prevents duplicate logging issues with PyTorch DDP.

    Args:
        name: Logger name for identification and hierarchy
        log_file: Path to the log file. If provided, logs will also be written to this file
                 (only for rank 0 process in distributed training)
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
                  Note: Only rank 0 process uses this level; others use ERROR level
        file_mode: File opening mode ('w' for write, 'a' for append)

    Returns:
        A configured logging.Logger instance

    Example:
        >>> logger = get_logger("my_model", "training.log", logging.DEBUG)
        >>> logger.info("Training started")
    """
    if file_mode not in ('w', 'a'):
        raise ValueError("file_mode must be either 'w' or 'a'")

    # Get or create logger instance
    logger = logging.getLogger(name)

    # Return existing logger if already initialized
    if name in logger_initialized:
        return logger

    # Check if parent logger is already initialized
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # Fix PyTorch DDP duplicate logging issue
    # Set root StreamHandler to ERROR level to prevent unwanted output from rank>0 processes
    for handler in logger.root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(logging.ERROR)

    # Initialize handlers list with stdout StreamHandler
    handlers = [logging.StreamHandler(sys.stdout)]

    # Determine process rank for distributed setup
    try:
        rank = dist.get_rank() if (dist.is_available()
                                   and dist.is_initialized()) else 0
    except Exception:
        rank = 0

    # Add FileHandler for rank 0 process if log_file is specified
    if rank == 0 and log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_file), file_mode))

    # Configure formatter and handlers
    formatter = ColorfulFormatter(
        fmt=
        '%(asctime)s - %(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Apply configuration to all handlers
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    # Set logger level based on rank
    logger.setLevel(log_level if rank == 0 else logging.ERROR)

    # Mark logger as initialized
    logger_initialized[name] = True

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:

            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name='llamatuner',
                        log_file=log_file,
                        log_level=log_level)

    return logger


def get_outdir(path: str, *paths, inc: bool = False) -> str:
    """Get the output directory. If the directory does not exist, it will be
    created. If `inc` is True, the directory will be incremented if the
    directory already exists.

    Args:
        path (str): The root path.
        *paths: The subdirectories.
        inc (bool, optional): Whether to increment the directory. Defaults to False.

    Returns:
        str: The output directory.
    """
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        return outdir
    elif inc:
        for count in range(1, 100):
            outdir_inc = f'{outdir}-{count}'
            if not os.path.exists(outdir_inc):
                os.makedirs(outdir_inc)
                return outdir_inc
        raise RuntimeError(
            'Failed to create unique output directory after 100 attempts')
    return outdir
