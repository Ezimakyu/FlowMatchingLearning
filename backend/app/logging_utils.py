from __future__ import annotations

import logging
import os

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str | None = None) -> None:
    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()

    if not root.handlers:
        logging.basicConfig(level=resolved, format=DEFAULT_LOG_FORMAT)
        return

    root.setLevel(resolved)
    for handler in root.handlers:
        handler.setLevel(resolved)


__all__ = ["configure_logging"]
