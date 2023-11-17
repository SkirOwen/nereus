from __future__ import annotations

import os
import shutil

from datetime import datetime

from tqdm import tqdm

from nereus import logger
from nereus.utils.downloader import downloader
from nereus.utils.directories import get_udash_dir

URL = "https://hs.pangaea.de/Projects/UDASH/UDASH.zip"


def download_udash(url: str, override: bool = False) -> None:
	udash_dir = get_udash_dir()
	downloader(url, udash_dir, override=override)


def _extract_udash(file: str) -> None:
	udash_dir = get_udash_dir()
	logger.debug("Extracting UDASH.")

	shutil.unpack_archive(
		filename=os.path.join(udash_dir, "UDASH.zip"),
		extract_dir=udash_dir,
		format="zip"
	)


def _udash_fileparser(filepath: str):
	lines = []
	with open(filepath, "r") as f:
		for line in f:
			lines.append(line)

	keys = []
	for v in lines[0]:
		if "Temp" in v:
			v = "Temp[C]"
		elif v == "QF":
			v = f"QF_{keys[-1]}"
		keys.append(v)

	data = {key: [] for key in keys}

	for line in tqdm(lines[1:]):
		for i, v in enumerate(line):
			key = keys[i]
			if key == "yyyy-mm-ddThh:mm":
				v = datetime.fromisoformat(v)
			elif v.replace('.', '', 1).isdigit():  # Check if the value is numeric
				if '.' in v:
					v = float(v)  # Convert to float if decimal
				else:
					v = int(v)  # Convert to int if whole number

			elif v.lstrip('-').replace('.', '', 1).isdigit():  # Check for negative numbers
				if '.' in v:
					v = float(v)
				else:
					v = int(v)
			data[key].append(v)



def main():
	download_udash(url=URL)


if __name__ == "__main__":
	main()
