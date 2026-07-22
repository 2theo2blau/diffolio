"""Small shared helpers: logging setup and JSON IO."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def setup_logging(level: str | int = "INFO") -> None:
    """Configure root logging once, idempotently."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt="%H:%M:%S")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False, default=str)
        fh.write("\n")
    tmp.replace(path)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
