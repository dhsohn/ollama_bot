"""Structured logging setup built on top of structlog."""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import structlog


def setup_logging(log_level: str = "INFO", log_dir: str | None = None) -> None:
    """Configure application-wide logging.

    Args:
        log_level: Log level such as DEBUG, INFO, WARNING, or ERROR.
        log_dir: Log file directory. If None, file logging is disabled.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    # Stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(stdout_handler)

    # File handler with daily rotation
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            filename=str(log_path / "app.log"),
            when="midnight",
            interval=1,
            backupCount=14,
            encoding="utf-8",
            utc=True,
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        file_handler.suffix = "%Y%m%d"
        root_logger.addHandler(file_handler)

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if log_dir:
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=False,
        )
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
        )
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)
    else:
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a logger scoped to a module."""
    return structlog.get_logger(name)
