import logging
import sys
from typing import Optional


def get_logger(name: Optional[str] = None, level: int | str = logging.INFO) -> logging.Logger:
    """Return a configured logger.

    - Adds a StreamHandler to stderr with a concise format.
    - Does not duplicate handlers if called multiple times.
    """
    logger = logging.getLogger(name if name else "Project")

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(stream=sys.stderr)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger

