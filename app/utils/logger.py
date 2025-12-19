"""
Logging utilities for the video translation pipeline.
Provides structured logging with consistent formatting.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from app.config import config


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Set up and configure a logger with console and optional file output.

    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file for persistent logging

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Use config level if not specified
    log_level = level or config.LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper()))

    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    # Console handler with color support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Formatter with timestamp and context
    formatter = logging.Formatter(
        config.LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler for persistent logs
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with standard configuration.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Configured logger instance
    """
    return setup_logger(name)


class LoggerMixin:
    """
    Mixin class to add logging capability to any class.
    Automatically creates a logger using the class name.
    """

    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for this class."""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger
