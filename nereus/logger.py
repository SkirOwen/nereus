import logging

from rich.logging import RichHandler

from rich.console import Console

console = Console()

def make_logger(level: str = "INFO") -> logging.Logger:
	FORMAT = "%(message)s"
	logging.basicConfig(
		level=logging.WARNING,
		format=FORMAT,
		datefmt="[%X]",
		handlers=[RichHandler(console=console)],
	)

	logger = logging.getLogger("nereus")
	logger.setLevel(level)
	return logger
