from __future__ import annotations

import logging

# This needs to be at the top as other files depend on it
from .logger import make_logger


__all__ = [
	"logger",
	"cluster",
	"plots",
	"processing",
]


logger: logging.Logger

logger = make_logger(
	level="INFO"
)

