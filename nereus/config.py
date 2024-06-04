import importlib
import inspect
import os

from rich.console import Console
console = Console()


def get_nereus_dir() -> str:
	rdp_module = importlib.import_module("nereus")
	rdp_dir = os.path.dirname(inspect.getabsfile(rdp_module))
	return os.path.abspath(os.path.join(rdp_dir, ".."))
