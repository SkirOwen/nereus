from __future__ import annotations


def convert_bytes(size: float) -> str:
	"""Function to convert bytes into a human-readable format
	https://stackoverflow.com/a/59174649/9931399

	Parameters
	----------
	size : float
		size in bytes to be converted.

	Returns
	-------
	str
		string of the converted size.
	"""
	for x in ['bytes', 'KiB', 'MiB', 'GiB', 'TiB']:
		if size < 1024.0:
			return f"{size:3.2f} {x}"
		size /= 1024.0
	return str(size)


def str2num(s: str) -> int | float | str:
	"""
	Convert a string to an int or a float.
	If the string is an integer number then convert to int.
	If the string is a decimal then convert to float.
	"""
	# is_number = s.lstrip('-').replace('.', '', 1).isdigit()
	# if is_number and not (s.startswith("0") and s != "0"):
	# 	if '.' in s:
	# 		s = float(s)
	# 	else:
	# 		s = int(s)
	try:
		s = float(s)
		if s.is_integer():
			s = int(s)
	except ValueError:
		pass
	return s
