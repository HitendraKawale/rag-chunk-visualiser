from __future__ import annotations

import logging
import sys

from rag_chunk_visualizer.core.config import settings

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str | None = None) -> None:
    resolved_level = (level or settings.log_level).upper()

    logging.basicConfig(
        level=getattr(logging, resolved_level, logging.INFO),
        format=LOG_FORMAT,
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
