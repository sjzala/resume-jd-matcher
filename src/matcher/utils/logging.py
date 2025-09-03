from __future__ import annotations

import sys

from loguru import logger


def configure(level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss} | {level:<7} | {message}")
