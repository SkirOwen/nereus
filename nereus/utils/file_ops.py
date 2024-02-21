from __future__ import annotations

import os
import hashlib


def guarantee_existence(path: str) -> str:
	"""Function to guarantee the existence of a path, and returns its absolute path.

	Parameters
	----------
	path : str
		Path (in str) to guarantee the existence.

	Returns
	-------
	str
		The absolute path.
	"""
	if not os.path.exists(path):
		os.makedirs(path)
	return os.path.abspath(path)


def calculate_md5(file_path: str) -> str:
	"""
	Calculate the MD5 checksum of a file.

	Parameters
	----------
	file_path : str
		The path to the file for which the MD5 checksum is to be calculated.

	Returns
	-------
	str
		The MD5 checksum of the file.
	"""
	hash_md5 = hashlib.md5()
	with open(file_path, "rb") as f:
		for chunk in iter(lambda: f.read(4096), b""):
			hash_md5.update(chunk)
	return hash_md5.hexdigest()


def create_cache_filename(name: str, ext: str, *args):
	combined_args = "_".join(map(str, args))

	hash_object = hashlib.md5(combined_args.encode())
	hash_string = hash_object.hexdigest()
	return f"name_{hash_string}.{ext}"
