from __future__ import annotations

import hashlib
import json
import os

from nereus.config import get_nereus_dir


CACHE_MAPPING = os.path.join(get_nereus_dir(), "nereus", "cache_mapping")
# TODO: guarantee existence of the CACHE_MAPPING

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


def create_cache_filename(name: str, **kwargs) -> str:
	combined_string = f"{name}_" + "_".join(f"{key}={value}" for key, value in kwargs.items())

	hash_object = hashlib.md5(combined_string.encode())
	hash_string = hash_object.hexdigest()

	# Check if the hash already exists in the cache mapping file
	if retrieve_processing_info(hash_string) is None:
		# Save the mapping
		with open(CACHE_MAPPING, "a", encoding="utf-8") as file:
			file.write(json.dumps({hash_string: kwargs}) + "\n")

	return f"name_{hash_string}"


def retrieve_processing_info(hash_string) -> str | None:
	with open(CACHE_MAPPING, "r", encoding="utf-8") as file:
		for line in file:
			mapping = json.loads(line)
			if hash_string in mapping:
				return mapping[hash_string]
	return None
