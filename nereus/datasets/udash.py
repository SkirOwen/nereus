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

	data = {key: [] for key in lines[0]}

	for line in tqdm(lines[1:]):
		for i, v in enumerate(line):
			key = lines[0][i]
			if key == "yyyy-mm-ddThh:mm":
				v = datetime.fromisoformat(v)
			elif v.isdigit():
				v = int(v)
			else:
				try:
					v = float(v)
				except ValueError:
					continue
			data[key].append(v)



def main():
	download_udash(url=URL)


if __name__ == "__main__":
	main()
