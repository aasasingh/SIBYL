"""Structured logging via structlog. Call `configure_logging()` once at startup."""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(*, json_output: bool = False, level: int = logging.INFO) -> None:
    logging.basicConfig(stream=sys.stderr, level=level, format="%(message)s")
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
