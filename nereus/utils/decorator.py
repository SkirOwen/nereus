from __future__ import annotations

import functools
import psutil

from nereus.utils.simple_functions import convert_bytes


def memory_usage(func):
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		# Get memory usage before the function call
		process = psutil.Process()
		mem_before = process.memory_info().rss

		# Execute the function
		result = func(*args, **kwargs)

		# Get memory usage after the function call
		mem_after = process.memory_info().rss

		print(f"Memory used by '{func.__name__}': {convert_bytes(mem_after - mem_before)}")
		return result

	return wrapper
