"""Logging configuration for kalmanbox.

Uses structlog if available, falls back to stdlib logging.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

_LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"

_has_structlog = False
try:
    import structlog

    _has_structlog = True
except ImportError:
    pass

if TYPE_CHECKING:
    import structlog


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the kalmanbox namespace.

    Parameters
    ----------
    name : str
        Logger name (will be prefixed with 'kalmanbox.').

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(f"kalmanbox.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
    return logger


def configure_logging(level: str = "WARNING", use_structlog: bool = True) -> None:
    """Configure kalmanbox logging.

    Parameters
    ----------
    level : str
        Log level: DEBUG, INFO, WARNING, ERROR.
    use_structlog : bool
        Use structlog if available (default True).
    """
    numeric_level = getattr(logging, level.upper(), logging.WARNING)
    root = logging.getLogger("kalmanbox")
    root.setLevel(numeric_level)

    if use_structlog and _has_structlog:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
