import logging

from rich.logging import RichHandler


def make_logger(level: str = "INFO") -> logging.Logger:
	FORMAT = "%(message)s"
	logging.basicConfig(
		level=logging.WARNING,
		format=FORMAT,
		datefmt="[%X]",
		handlers=[RichHandler()],
	)

	logger = logging.getLogger("nereus")
	logger.setLevel(level)
	return logger
