from __future__ import annotations

import ast
import asyncio
import datetime
import functools
import glob
import os.path
import re
import shutil
import ssl
import urllib.request

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from urllib.parse import urljoin

import aiohttp
import numpy as np
import pandas as pd
import xarray as xr

from tqdm import tqdm

from nereus import logger
from nereus.utils.directories import get_itp_cache_dir, get_itp_dir, get_itp_extracted_dir
from nereus.utils.downloader import downloader
from nereus.utils.file_ops import calculate_md5
from nereus.utils.iterable_ops import skipwise


URL = "https://scienceweb.whoi.edu/itp/data/"
MD5_URL = "https://scienceweb.whoi.edu/itp-md5sums/MD5SUMS"

COL_ITP = [
	"pressure(dbar)",
	"temperature(C)",
	"salinity",
	"nobs",
	"file",
	"east(cm/s)",
	"north(cm/s)",
	"vert(cm/s)",
	"nacm",
	"dissolved_oxygen",
	"CDOM(ppb)",
	"PAR(uE/m^2/s)",
	"turbidity(/m/sr)x10^4",
	"chlorophyll-a(ug/l)",
	"dissolved_oxygen(umol/kg)",
	"turbidity(e-4)",
	"chlorophyll_a(ug/l)",
	"nbio",
]

COL_META = ["file", "source", "ITP", "profile", "year", "day", "longitude(E+)", "latitude(N+)", "ndepths", "time"]


RENAME_COL = {
	"pressure(dbar)": "pres",
	"temperature(C)": "temp",
	"salinity": "sal",
	"east(cm/s)": "east",
	"north(cm/s)": "north",
	"vert(cm/s)": "vert",
	"CDOM(ppb)": "CDOM",
	# 'PAR(uE/m^2/s)':,
	# 'turbidity(/m/sr)x10^4',
	# 'chlorophyll-a(ug/l)',
	"dissolved_oxygen(umol/kg)": "dis_oxy",
	# 'turbidity(e-4)',
	# 'chlorophyll_a(ug/l)',
	"longitude(e+)": "lon",
	"latitude(n+)": "lat",
	# 'ndepths',
	# 'time'
}


async def async_get_filenames_from_url(url: str) -> list[str]:
	"""
	Asynchronously gets the filename list for files ending with 'final.zip' from an archive-like URL.

	Parameters
	----------
	url : str
		The URL from which to scrape the filenames.

	Returns
	-------
	list[str]
		A list of filenames ending with 'final.zip'. If no such filenames are found,
		an empty list is returned. If an error occurs, an empty list is returned and the error
		is logged.
	"""
	async with aiohttp.ClientSession() as session:
		try:
			async with session.get(url, headers={
				"User-Agent":
					"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
					"AppleWebKit/537.36 (KHTML, like Gecko) "
					"Chrome/103.0.0.0 Safari/537.36"
			}) as response:
				html_content = await response.text()

			lines = html_content.split("\n")

			file_urls = []
			directories_names = []

			for line in lines:
				if "href=" in line:
					start_idx = line.find('href="') + 6
					end_idx = line.find('"', start_idx)
					if start_idx != -1 and end_idx != -1:
						file_name = line[start_idx:end_idx]

						if not file_name.startswith("/") and file_name.endswith("/"):
							directories_names.append(file_name)
						elif file_name.endswith("final.zip"):
							file_urls.append(urljoin(url, file_name))

			# Async calls for directories
			if directories_names:
				pbar = tqdm(total=len(directories_names), desc="Finding ITPs")
				tasks = []
				for directory in directories_names:
					task = asyncio.create_task(async_get_filenames_from_url(urljoin(url, directory)))
					task.add_done_callback(lambda p: pbar.update())
					tasks.append(task)

				directory_files = await asyncio.gather(*tasks)
				pbar.close()

				for files in directory_files:
					file_urls.extend(files)

			return file_urls

		except (aiohttp.ClientError, aiohttp.InvalidURL, aiohttp.ClientResponseError, asyncio.TimeoutError) as e:
			logger.error(f"Error accessing {url}: {e}\n" f"Try with '_get_filenames_from_url'")
			return []


def _get_filenames_from_url(url: str) -> list[str]:
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
		"user-agent",
		"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
		"AppleWebKit/537.36 (KHTML, like Gecko) "
		"Chrome/103.0.0.0 Safari/537.36",
	)
	response = urllib.request.urlopen(req)
	html_content = response.read().decode("utf-8")

	lines = html_content.split("\n")

	file_urls = []
	directories_names = []
	for line in lines:
		if "href=" in line:
			start_idx = line.find('href="') + 6  # Find the start index of the filename
			end_idx = line.find('"', start_idx)  # Find the end index of the filename
			if start_idx != -1 and end_idx != -1:  # Check that both indices were found
				file_name = line[start_idx:end_idx]

				if not (file_name.startswith("/")) and file_name.endswith("/"):
					directories_names.append(file_name)

				if file_name.endswith("final.zip"):
					file_urls.append(urljoin(url, file_name))

	for directory in tqdm(directories_names):
		directory_url = urljoin(url, directory)
		file_urls.extend(_get_filenames_from_url(directory_url))

	return file_urls


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

	hash_md5 = calculate_md5(md5sum_filepath)
	if not (hash_md5 == "092cc1dccdd6fbbc0521273e2133fa11"):
		logger.info("The md5 of the md5 file did not match.")

	with open(md5sum_filepath, "r", encoding="utf-8") as f:
		lines = f.readlines()

	md5_dict = {
		filename: value for line in lines for filename, value in [line.split()]
	}

	return md5_dict


def download_itp(files_urls: None | list[str] = None, override: bool = False) -> None:
	"""
	Downloads files with the extension 'final.zip' from the specified main URL.

	Parameters
	----------
	main_url : str
		The main URL where the files are hosted.
	files_urls : None | list[str], optional
		A list of filenames to download. If `None`, the function will scrape the filenames
		using `get_filenames_from_url` function. Default is `None`.
	override : bool, optional
		Whether to override existing files with the same name. Default is `False`.
	"""
	if files_urls is None:
		files_urls = asyncio.run(async_get_filenames_from_url(URL))

	itp_dir = get_itp_dir()

	downloader(files_urls, itp_dir, override=override)


def _extract_itp(file: str, target_directory: None | str = None) -> None:
	"""

	Parameters
	----------
	file : str
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
		for itp in tqdm(all_itps):
			itp_filepath = os.path.join(itp_dir, itp)
			pool.submit(_extract_itp, itp_filepath, target_dir)
		logger.info("All ITPs have been extracted.")


def itp_parser(
		filepath: str,
		filtering: bool = True,
		nbr_filter: int = 2,
		low_filter: float = 10.0,
		high_filter: float = 750.0,
) -> tuple[pd.DataFrame, dict] | None:
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
	filtering : bool, optional
	nbr_filter : int, optional
	low_filter : float, optional
	high_filter : float, optional

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

	if filtering:
		if len(lines) <= nbr_filter + 5:
			return None

		if float(lines[3].split()[0]) > low_filter:
			return None

		if float(lines[-2].split()[0]) < high_filter:
			return None

	# the header of the metadata is in two parts separated by a colon
	# the left part follows this:
	# %NAME VALUE, NAME VALUE, ..., NAME VALUE
	# the right part this one:
	# NAME NAME NAME NAME
	instrument_info, attribute_names = lines[0].split(":")

	# Line 0 and 1 stores the metadata
	# Using re to remove the "%", ":", "," character form the string
	instrument_info = re.sub(r"[%,]", "", instrument_info).lower().split()

	metadata.update(skipwise(instrument_info, step=2))

	attribute_names = re.sub(r"[%,]", "", attribute_names).lower().split()
	metadata_values = list(map(ast.literal_eval, lines[1].split()))
	# casting the float values of the metadata to floats as they are stored in str

	metadata.update(zip(attribute_names, metadata_values))

	metadata["time"] = (
		datetime.datetime(year=metadata["year"], month=1, day=1) +
		datetime.timedelta(days=metadata["day"] - 1)  # -1 because Jan 1st is day 1.0000
	)

	# The name of the variables are stored on line 2
	data_names = lines[2][1:].split()
	# name.lower().split("(")[0] would remove the unit and parenthesis
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

	return pd.DataFrame(data), metadata


def parser_all_itp(limit: int = None, **kwargs) -> tuple:
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

	with Pool() as pool:
		for results in tqdm(
			pool.imap(functools.partial(itp_parser, **kwargs), files), total=len(files), desc="Parsing itps"
		):
			if results is not None:
				data, metadata = results
				itps.append(data)
				metadatas.append(metadata)

	metadata = pd.DataFrame(metadatas)
	metadata = metadata.set_index("file")
	logger.info(f"{len(itps)} itps matching filter")
	return itps, metadata


# def itps_to_df(save_df: bool = True, regenerate: bool = False):
# 	""""""
# 	itps_filepath = os.path.join(get_itp_dir(), "itps.parquet")
# 	metadata_filepath = os.path.join(get_itp_dir(), "metadata.csv")
#
# 	cache_exist = os.path.exists(itps_filepath) and os.path.exists(metadata_filepath)
#
# 	# TODO: backend option to choose pandas vs polars
# 	if regenerate or not cache_exist:
# 		itps, metadatas = parser_all_itp()
#
# 		df_metadatas = pl.DataFrame(metadatas)
# 		logger.info("Converting ITPs to dataframe")
# 		df_itps = pl.concat([pl.DataFrame(itp) for itp in tqdm(itps, desc="Itps")], how="diagonal")
# 		if save_df:
# 			logger.info("Saving to file")
# 			df_itps.write_parquet(itps_filepath)
# 			df_metadatas.write_csv(metadata_filepath)
#
# 	else:
# 		df_itps = pd.read_parquet(itps_filepath)
# 		df_metadatas = pd.read_csv(metadata_filepath)
#
# 	return df_itps, df_metadatas


# def load_itp(regenerate: bool = False, join: bool = False):
# 	itps_filepath = os.path.join(get_itp_dir(), "itps.parquet")
# 	metadata_filepath = os.path.join(get_itp_dir(), "metadata.csv")
#
# 	cache_exist = os.path.exists(itps_filepath) and os.path.exists(metadata_filepath)
#
# 	if regenerate or not cache_exist:
# 		itps_to_df()
#
# 	df_itps = pd.read_parquet(itps_filepath)
# 	df_metadatas = pd.read_csv(metadata_filepath)
# 	if join:
# 		df_metadatas = df_metadatas.set_index("file")
# 		return df_itps.join(df_metadatas, on="file")
# 	else:
# 		return df_itps, df_metadatas


def interp_itps(itp: pd.DataFrame, dims: list[str], x_inter, base_dim: str) -> pd.DataFrame:
	# This will take dims as y for interpolation
	# such as dims = f(base_dim)
	# then take x_inter, a range of point on which to interpolate
	# it will return?? a dict? a new df?
	# if it returns a df, should the input be a df?
	# Handle NaNs
	# Can remove the nans.
	# or just quickly to pd interp?
	# That could be a parameter

	x_inter = np.arange(10, 760, 10)
	interp_itp = {
		"file": np.full(x_inter, itp["file"].values[0]),  # So everything has the same length
		base_dim: x_inter,
	}

	for dim in dims:
		if dim in itp:
			interp_itp[dim] = np.interp(x_inter, itp[base_dim].values, itp[dim].values)
		else:
			interp_itp[dim] = np.full(x_inter.shape, np.nan)
	return pd.DataFrame(interp_itp)


def itps_to_xr(df_itps: pd.DataFrame) -> xr.Dataset:
	unique_coords = df_itps.drop_duplicates("file").set_index("file")[["lat", "lon", "time"]]
	df_itps.rename(columns={"file": "profile"}, inplace=True)
	df_itps.set_index(["profile", "pres"], inplace=True)

	ds = xr.Dataset.from_dataframe(df_itps)
	for coord in ["lat", "lon", "time"]:
		ds = ds.assign_coords({coord: ("profile", unique_coords[coord])})
	return ds


def preload_itp(clean_df=True, regen: bool = False, **kwargs):
	# check download
	# parse
	save_path = os.path.join(get_itp_cache_dir(), "itps_xr.nc")

	if not os.path.exists(save_path) or regen:
		parquet_cache = os.path.join(get_itp_cache_dir(), "itps_preprocessed.parquet")
		if not os.path.exists(parquet_cache) or regen:
			itps, metadatas = parser_all_itp(**kwargs)
			logger.info("Parsed")
			processed_itps = []

			for itp in tqdm(itps):
				new_itp = interp_itps(itp, **kwargs)
				processed_itps.append(new_itp)

			logger.info("Concat")
			df_itps = pd.concat(
				processed_itps, ignore_index=True, keys=metadatas.index.get_level_values("file").to_list()
			)

			logger.info("Join")
			df_itps = df_itps.join(metadatas, on="file")

			df_itps.rename(columns=RENAME_COL, inplace=True)
			if clean_df:
				logger.info("Clean df")
				df_itps.drop(["source", "year", "day", "profile", "itp"], axis=1, inplace=True)

			logger.info("Caching")
			df_itps.to_parquet(parquet_cache)
		else:
			df_itps = pd.read_parquet(parquet_cache)

		logger.info("Converting to xarray")
		ds = itps_to_xr(df_itps)

		logger.info("Saving xr")

		ds.to_netcdf(
			save_path,
			format="NETCDF4",
			engine="h5netcdf",
		)

	return save_path


def main():
	preload_itp(

	)
	# print(itps_path)
	# meta, itp = parser_all_itp(filtering=False)
	# print(len(meta))
	# print(len(itp))


if __name__ == "__main__":
	main()
