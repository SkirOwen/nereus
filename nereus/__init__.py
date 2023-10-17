import logging

from .logger import make_logger

__all__ = ["logger"]

logger: logging.Logger

logger = make_logger(
	level="INFO"
)

