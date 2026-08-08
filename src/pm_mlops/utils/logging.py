"""Shared logging setup so every script/module logs consistently."""

from __future__ import annotations

import sys

from loguru import logger

_CONFIGURED = False


def get_logger(name: str | None = None):
    """Return a configured loguru logger, configuring sinks only once.

    Parameters
    ----------
    name:
        Optional context name bound to the returned logger (e.g. the
        calling module), shown in every log line it emits.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        logger.remove()
        logger.add(
            sys.stderr,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{extra[context]}</cyan> - <level>{message}</level>"
            ),
            level="INFO",
        )
        logger.configure(extra={"context": "pm_mlops"})
        _CONFIGURED = True

    return logger.bind(context=name or "pm_mlops")
