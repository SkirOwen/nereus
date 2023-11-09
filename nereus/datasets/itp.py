from __future__ import annotations

import ast
import datetime
import concurrent.futures
import glob
import os.path
import re
import shutil
import ssl
import urllib.request

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import pandas as pd
import polars as pl

from tqdm import tqdm

from nereus import logger

from nereus.utils.downloader import downloader
from nereus.utils.file_ops import calculate_md5
from nereus.utils.directories import get_itp_dir, get_itp_extracted_dir
from nereus.utils.iterable_ops import skipwise

URL = "https://scienceweb.whoi.edu/itp/data/"
MD5_URL = "https://scienceweb.whoi.edu/itp-md5sums/MD5SUMS"


def get_filenames_from_url(url: str) -> list[str]:
	"""
	Gets the filename list for files ending with 'final.tar.Z' from an archive-like URL.

	Parameters
	----------
	url : str
		The URL from which to scrape the filenames.

	Returns
	-------
	list[str]
		A list of filenames ending with 'final.tar.Z'. If no such filenames are found,
		an empty list is returned. If an error occurs, an empty list is returned and the error
		is logged.
	"""

	ssl._create_default_https_context = ssl._create_unverified_context
	req = urllib.request.Request(url)
	req.add_header(
		'user-agent',
		'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'
	)
	response = urllib.request.urlopen(req)
	html_content = response.read().decode("utf-8")

	lines = html_content.split("\n")

	file_names = []
	for line in lines:
		if "final.zip" in line:
			start_idx = line.find('href="') + 6  # Find the start index of the filename
			end_idx = line.find('"', start_idx)  # Find the end index of the filename
			if start_idx != -1 and end_idx != -1:  # Check that both indices were found
				file_name = line[start_idx:end_idx]
				file_names.append(file_name)

	return file_names


def get_md5sum_dict() -> dict[str, str]:
	"""
	Function to download and return the md5 sums of the ipts as a dictionary.

	Returns
	-------
	dict,
		Keys are the name of the file and the value is their md5 check sum.
	"""
	md5sum_filepath = os.path.join(get_itp_dir(), "MD5SUMS")
	if not os.path.exists(md5sum_filepath):
		downloader([MD5_URL], get_itp_dir())

	hash = calculate_md5(md5sum_filepath)
	if not(hash == "67ecdfe4bac8a5fd277bdf67cb59d7b6"):
		logger.info("The md5 of the md5 file did not match.")

	with open(md5sum_filepath, "r") as f:
		lines = f.readlines()

	md5_dict = {
		filename: value for line in lines for filename, value in [line.split()]
	}

	return md5_dict


def download_itp(main_url: str, files: None | list[str] = None, override: bool = False) -> None:
	"""
	Downloads files with the extension 'final.tar.Z' from the specified main URL.

	Parameters
	----------
	main_url : str
		The main URL where the files are hosted.
	files : None | list[str], optional
		A list of filenames to download. If `None`, the function will scrape the filenames
		using `get_filenames_from_url` function. Default is `None`.
	override : bool, optional
		Whether to override existing files with the same name. Default is `False`.
	"""
	if files is None:
		files = get_filenames_from_url(main_url)

	itp_dir = get_itp_dir()

	urls = [main_url + f for f in files]
	downloader(urls, itp_dir, override=override)


def extract_itp(file: str, target_directory: None | str = None) -> None:
	"""

	Parameters
	----------
	file
	target_directory : str, optional
		Default will be ./data/itp/extracted

	Returns
	-------
	"""
	target_directory = get_itp_extracted_dir() if target_directory is None else target_directory
	shutil.unpack_archive(filename=file, extract_dir=target_directory, format="zip")


def extract_all_itps(itp_dir: str, target_dir: None | str = None):
	all_itps = [
		f for f in os.listdir(itp_dir)
		if f.endswith(".zip")
	]

	with ThreadPoolExecutor(max_workers=4) as pool:
		for task, itp in enumerate(tqdm(all_itps)):
			itp_filepath = os.path.join(itp_dir, itp)
			pool.submit(extract_itp, itp_filepath, target_dir)
		logger.info("All ITPs have been extracted.")


def itp_parser(filepath, progress_bar=None) -> tuple[dict, dict]:
	"""
	Parse data from an ITP file.

	This function reads data from the specified ITP file and extracts both metadata
	and data values.

	The ITP file format is assumed to have metadata on the first two lines, variable names on the third line,
	and data starting from the fourth line until the end.

	Parameters
	----------
	filepath : str
		The path to the ITP file to be parsed.

	Returns
	-------
	tuple of dict and dict
		A tuple containing two dictionaries:
			- The first dictionary contains the parsed data values, where keys are variable names
			and values are lists of corresponding data points.
			- The second dictionary contains the parsed metadata of the itp, saved in the same order.
	"""
	metadata = {
		"file": os.path.basename(filepath),
		"source": "ITP",
	}

	with open(filepath, "r") as f:
		lines = f.readlines()

	# the header of the metadata is in two parts separated by a colon
	# the left part follows this:
	# %NAME VALUE, NAME VALUE, ..., NAME VALUE
	# the right part this one:
	# NAME NAME NAME NAME
	instrument_info, attribute_names = lines[0].split(":")

	# Line 0 and 1 stores the metadata
	# Using re to remove the "%", ":", "," character form the string
	instrument_info = re.sub(r"[%,]", "", instrument_info).split()

	metadata.update(skipwise(instrument_info, step=2))

	attribute_names = re.sub(r"[%,]", "", attribute_names).split()
	metadata_values = list(map(ast.literal_eval, lines[1].split()))
	# casting the float values of the metadata to floats as they are stored in str

	metadata.update(zip(attribute_names, metadata_values))

	metadata["time"] = (
		datetime.datetime(year=metadata["year"], month=1, day=1) +
		datetime.timedelta(days=metadata["day"] - 1)    # -1 because Jan 1st is day 1.0000
	)

	# The name of the variables are stored on line 2
	data_names = lines[2][1:].split()
	data = {name: [] for name in data_names}

	# The data start at line 3
	# Line -1 is an eof tag, so ignoring it
	for line in lines[3:-1]:
		# values = list(map(ast.literal_eval, line.split()))
		values = np.fromstring(line, sep="\t")
		for name, val in zip(data_names, values):
			data[name].append(val)

	data["file"] = os.path.basename(filepath)
	if "nobs" in data_names:
		data["nobs"] = list(map(int, data["nobs"]))

	if progress_bar is not None:
		progress_bar.update(1)

	return data, metadata


def parser_all_itp(limit: int = None) -> tuple:
	all_files = glob.glob(os.path.join(get_itp_extracted_dir(), "*.dat"))
	micro_files = glob.glob(os.path.join(get_itp_extracted_dir(), "*micro*.dat"))
	sami_files = glob.glob(os.path.join(get_itp_extracted_dir(), "*sami*.dat"))
	files = list(set(all_files) - (set(micro_files) | set(sami_files)))
	logger.info(f"Found {len(files)} ITPs to parse")

	if limit is not None and limit <= len(files):
		logger.info(f"Only parsing {limit} files")
		files = files[:limit]

	metadatas = []
	itps = []
	# with ThreadPoolExecutor() as pool:
	# 	with tqdm(total=len(files)) as progress_bar:
	# 		futures = {pool.submit(itp_parser, file, progress_bar): file for file in files}
	# 		for future in concurrent.futures.as_completed(futures):
	# 			data, metadata = future.result()
	# 			itps.append(data)
	# 			metadatas.append(metadata)

	with Pool() as pool:
		for data, metadata in tqdm(pool.imap(itp_parser, files), total=len(files), desc="Parsing itps"):
			itps.append(data)
			metadatas.append(metadata)

	return itps, metadatas


def itps_to_df(save_df: bool = True, regenerate: bool = False):
	""""""
	itps_filepath = os.path.join(get_itp_dir(), "itps.parquet")
	metadata_filepath = os.path.join(get_itp_dir(), "metadata.csv")

	cache_exist = os.path.exists(itps_filepath) and os.path.exists(metadata_filepath)

	# TODO: backend option to choose pandas vs polars
	if regenerate or not cache_exist:
		itps, metadatas = parser_all_itp()

		df_metadatas = pl.DataFrame(metadatas)
		logger.info("Converting ITPs to dataframe")
		df_itps = pl.concat([pl.DataFrame(itp) for itp in tqdm(itps, desc="Itps")], how="diagonal")
		if save_df:
			logger.info("Saving to file")
			df_itps.write_parquet(itps_filepath)
			df_metadatas.write_csv(metadata_filepath)

	else:
		df_itps = pl.read_parquet(itps_filepath)
		df_metadatas = pl.read_csv(metadata_filepath)

	return df_itps, df_metadatas


def query_from_metadata(query: str) -> list:
	# load metadata file
	# find file names to load
	# delay load
	pass


def main():
	itps_to_df()


if __name__ == "__main__":
	main()
