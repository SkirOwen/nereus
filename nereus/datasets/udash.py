from __future__ import annotations

import glob
import os
import shutil

from datetime import datetime, date
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

from nereus import logger
from nereus.utils.downloader import downloader
from nereus.utils.directories import get_udash_dir, get_udash_extracted_dir
from nereus.utils.simple_functions import str2num

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
	for v in lines[0].split():
		if "Temp" in v:
			v = "Temp[C]"
		elif v == "QF":
			v = f"QF_{keys[-1]}"
		keys.append(v)

	data = {key: [] for key in keys}

	for line in (lines[1:]):
		for i, v in enumerate(line.split()):
			key = keys[i]
			if key == "yyyy-mm-ddThh:mm":
				if "99:99" in v:
					v = date.fromisoformat(v[:-6])
				else:
					try:
						v = datetime.fromisoformat(v)
					except ValueError:
						continue
			else:
				v = str2num(v)
			data[key].append(v)
	return data


def parse_udash(files: None | list[str] = None):
	udash_extracted_dir = get_udash_extracted_dir()
	if files is None:
		files = glob.glob(os.path.join(udash_extracted_dir, "ArcticOcean_*.txt"))
	logger.info(f"{len(files)} udash files to parse.")

	udash = []
	# for f in tqdm(files):
	# 	udash.append(_udash_fileparser(f))

	with Pool() as pool:
		for data in tqdm(pool.imap(_udash_fileparser, files), total=len(files), desc="Parsing udash"):
			udash.append(data)

	for u in udash:
		df = pd.concat(pd.DataFrame(u))

	df.to_parquet(os.path.join(get_udash_dir(), "udash.parquet"))



def main():
	parse_udash()


if __name__ == "__main__":
	main()
