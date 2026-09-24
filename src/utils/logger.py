"""Project logging setup."""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str = "lv_xai", log_file: str | Path | None = None) -> logging.Logger:
    """Create or retrieve a consistently formatted project logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
    if log_file is not None and not any(
        isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == Path(log_file).resolve()
        for handler in logger.handlers
    ):
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(file_handler)
    return logger

