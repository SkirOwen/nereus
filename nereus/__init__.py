from __future__ import annotations

import logging

# This needs to be at the top as other files depend on it
from .logger import make_logger
from .config import console


__all__ = [
	"console",
	"logger",
	"cluster",
	"datasets",
	"plots",
	"processing",
]


logger: logging.Logger

logger = make_logger(
	level="INFO",
)
