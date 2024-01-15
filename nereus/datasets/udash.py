from __future__ import annotations

import glob
import os
import shutil

from datetime import datetime, date
from multiprocessing import Pool

import pandas as pd
import polars as pl
from tqdm import tqdm

from nereus import logger
from nereus.utils.downloader import downloader
from nereus.utils.directories import get_udash_dir, get_udash_extracted_dir
from nereus.utils.simple_functions import str2num

URL = "https://hs.pangaea.de/Projects/UDASH/UDASH.zip"

UDASH_COLUMN_TYPE = {
	'Prof_no': int,
	'Cruise': str,
	'Station': str,
	'Platform': str,
	'Type': str,
	'yyyy-mm-ddThh:mm': datetime,
	'Longitude_[deg]': float,
	'Latitude_[deg]': float,
	'Pressure_[dbar]': float,
	'Depth_[m]': float,
	'QF_Depth_[m]': int,
	'Temp[C]': float,
	'QF_Temp[C]': int,
	'Salinity_[psu]': float,
	'QF_Salinity_[psu]': int,
	'Source': str,
	'DOI': str,
	'WOD-Cruise-ID': str,
	'WOD-Cast-ID': str,
}


def download_udash(url: str, override: bool = False) -> None:
	udash_dir = get_udash_dir()
	downloader(url, udash_dir, override=override)


def _extract_udash(file: None | str = None) -> None:
	udash_dir = get_udash_dir()
	logger.info("Extracting UDASH.")

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
						pass
			data[key].append(v)
	return data


def parse_udash(files: None | list[str] = None, files_nbr: None | int = None):
	udash_extracted_dir = get_udash_extracted_dir()
	if files is None:
		files = glob.glob(os. path.join(udash_extracted_dir, "ArcticOcean_*.txt"))

	if files_nbr is not None:
		files = files[:files_nbr]

	logger.info(f"{len(files)} udash files to parse.")

	udash = []
	for f in tqdm(files):
		dd = _udash_fileparser(f)
		udash.append(pl.DataFrame(dd))
		# print(pd.DataFrame(dd).info())

	# with Pool() as pool:
	# 	for data in tqdm(pool.imap(_udash_fileparser, files), total=len(files), desc="Parsing udash"):
	# 		udash.append(pl.DataFrame(data))
	logger.info("Merging dataframe")
	df = pl.concat(udash, how="diagonal")
	df = convert_type(df)

	df.write_parquet(os.path.join(get_udash_dir(), "udash.parquet"))
	return df


def convert_type(df: pl.DataFrame):
	col = df.columns
	# TODO: can just hard-code the column I want to change, instead of getting the column from the df
	# Though, what if one column does not exist, it would return an error
	# now still the same issue but it just assumed it is matching with UDASH_COLUMN_TYPE
	for c, c_type in zip(col, UDASH_COLUMN_TYPE.values()):
		print(c_type)
		if c != "yyyy-mm-ddThh:mm":
			df = df.with_columns(pl.col(c).cast(c_type))
	return df


def load_udash(regenerate: bool = False):
	udash_filepath = os.path.join(get_udash_dir(), "udash.parquet")

	if regenerate or not os.path.exists(udash_filepath):
		parse_udash()

	df = pd.read_parquet(udash_filepath)
	return df


def main():
	# download_udash(URL)
	# _extract_udash()
	parse_udash()


if __name__ == "__main__":
	main()
