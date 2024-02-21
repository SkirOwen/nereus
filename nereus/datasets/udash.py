from __future__ import annotations

import glob
import os
import shutil

from datetime import datetime, date
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from tqdm import tqdm
from rich.progress import track

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
	# 'yyyy-mm-ddThh:mm': datetime,
	'Longitude_[deg]': float,
	'Latitude_[deg]': float,
	'Pressure_[dbar]': float,
	'Depth_[m]': float,
	'QF_Depth_[m]': int,
	'Temp_[C]': float,
	'QF_Temp_[C]': int,
	'Salinity_[psu]': float,
	'QF_Salinity_[psu]': int,
	'Source': str,
	'DOI': str,
	'WOD-Cruise-ID': str,
	'WOD-Cast-ID': str,
}

rename_col = {
	# 'Prof_no': int,
	# 'Cruise': str,
	# 'Station': str,
	# 'Platform': str,
	# 'Type': str,
	'yyyy-mm-ddThh:mm': "time",
	'Longitude_[deg]':  "lon",
	'Latitude_[deg]':   "lat",
	'Pressure_[dbar]':  "press",
	'Depth_[m]':        "depth",
	# 'QF_Depth_[m]': int,
	'Temp_[C]':          "temp",
	# 'QF_Temp_[C]': int,
	'Salinity_[psu]':   "sal",
	# 'QF_Salinity_[psu]': int,
	# 'Source': str,
	# 'DOI': str,
	# 'WOD-Cruise-ID': str,
	# 'WOD-Cast-ID': str,
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
			v = "Temp_[C]"
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

	data = pd.DataFrame(data)
	data = data.astype(UDASH_COLUMN_TYPE)
	data.replace(-999, np.nan, inplace=True)
	data.rename(columns=rename_col, inplace=True)
	data["time"] = pd.to_datetime(data["time"])
	return data


def parse_all_udash(files: None | list[str] = None, files_nbr: None | int = None, cache: str = "xarray"):
	udash_extracted_dir = get_udash_extracted_dir()
	if files is None:
		files = glob.glob(os. path.join(udash_extracted_dir, "ArcticOcean_*.txt"))

	if files_nbr is not None:
		files = files[:files_nbr]

	logger.info(f"{len(files)} udash files to parse.")

	udash = []
	for f in track(files, description="UDASH files parsing"):
		df = _udash_fileparser(f)
		udash.append(df)

	udash = pd.concat(udash, ignore_index=True)

	if filter:
		# Use the filter method to apply the conditions to each group
		udash_range = udash.groupby('file').filter(
			partial(filter_groups, dim=dim, low=low, high=high, min_nobs=min_nobs)
		)


def filter_groups(group, dim, low, high, min_nobs):
	mask = (
		group[dim].max() >= high and
		group[dim].min() <= low and
		len(group[dim]) > min_nobs
	)
	return mask


def parse_udash(files: None | list[str] = None, files_nbr: None | int = None, cache: str = "xarray"):
	udash_extracted_dir = get_udash_extracted_dir()
	if files is None:
		files = glob.glob(os. path.join(udash_extracted_dir, "ArcticOcean_*.txt"))

	if files_nbr is not None:
		files = files[:files_nbr]

	logger.info(f"{len(files)} udash files to parse.")

	udash = []
	for f in track(files, description="UDASH files parsing"):
		dd = _udash_fileparser(f)
		udash.append(pl.DataFrame(dd))
		# print(pd.DataFrame(dd).info())

	# with Pool() as pool:
	# 	for data in tqdm(pool.imap(_udash_fileparser, files), total=len(files), desc="Parsing udash"):
	# 		udash.append(pl.DataFrame(data))
	logger.info("Merging dataframe")
	df = pl.concat(udash, how="diagonal")
	df = convert_type(df)

	if cache == "parqet":
		df.write_parquet(os.path.join(get_udash_dir(), "udash.parquet"))
	# if cache == "xarray":
	# 	df = udash_xr(df, save=True)
	return df


def convert_type(df: pl.DataFrame) -> pd.DataFrame:
	col = df.columns
	# TODO: can just hard-code the column I want to change, instead of getting the column from the df
	# Though, what if one column does not exist, it would return an error
	# now still the same issue but it just assumed it is matching with UDASH_COLUMN_TYPE
	for c, c_type in zip(col, UDASH_COLUMN_TYPE.values()):
		print(c_type)
		if c != "yyyy-mm-ddThh:mm":
			df = df.with_columns(pl.col(c).cast(c_type))
	return df


def udash_xr(df: pd.DataFrame, save: bool = True) -> xr.Dataset:
	logger.info("Converting to xarray")
	ds = xr.Dataset.from_dataframe(df)
	ds = ds.rename({
		"yyyy-mm-ddThh:mm": "time",
		"Longitude_[deg]": "lon",
		"Latitude_[deg]": "lat",
	})
	ds = ds.set_coords("time")
	ds = ds.swap_dims({"index": "time"})
	ds = ds.drop_vars("index")
	ds = ds.set_coords(["lon", "lat"])
	logger.info("Converted!")
	if save:
		logger.info("Caching")
		ds.to_netcdf(os.path.join(get_udash_dir(), "udash.nc"))
	return ds


def load_udash(
		cache: str = "xarray",
		drop_argo: bool = False,
		drop_itp: bool = False,
		regenerate: bool = False,
		file: str | None = "udash_no_itp_argo.nc",
	) -> pd.DataFrame | xr.Dataset:

	file_ext = "nc" if cache == "xarray" else "parquet"
	file = f"udash.{file_ext}" if file is None else file

	udash_filepath = os.path.join(get_udash_dir(), file)
	udash_parquet_filepath = os.path.join(get_udash_dir(), "udash.parquet")
	xr_flag = False

	if regenerate or not os.path.exists(udash_filepath):
		if os.path.exists(udash_parquet_filepath):
			udash = udash_xr(pd.read_parquet(udash_parquet_filepath))
			xr_flag = True
		else:
			parse_udash(cache=cache)

	if cache == "xarray" and not xr_flag:
		udash = xr.open_dataset(udash_filepath)
		if drop_itp:
			udash = udash.where(np.logical_not(udash.Cruise.str.contains("itp")))
		if drop_argo:
			udash = udash.where(udash.Source != "argo", drop=True)
	if cache == "parquet":
		udash = pd.read_parquet(udash_filepath)
	return udash


def interp_udash(udash: pd.DataFrame, dims: list[str], x_inter, base_dim: str, **kwargs) -> pd.DataFrame:
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
		"file": udash["file"].values[:len(x_inter)],  # So everything has the same length
		base_dim: x_inter
	}

	for dim in dims:
		if dim in udash:
			interp_itp[dim] = np.interp(x_inter, udash[base_dim].values, udash[dim].values)
		else:
			interp_itp[dim] = np.full(x_inter.shape, np.nan)
	return pd.DataFrame(interp_itp)


def preload_itp(**kwargs):
	# check download
	# parse
	udash = parse_all_udash()
	logger.info("Parsed")
	processed_itps = []

	for itp in tqdm(udash):
		new_itp = interp_udash(itp, **kwargs)
		processed_itps.append(new_itp)

	logger.info("Concat")
	df_itps = pd.concat(processed_itps, ignore_index=True)

	df_itps.rename(columns=rename_col, inplace=True)

	logger.info("Caching")
	df_itps.to_parquet(os.path.join(get_udash_dir(), "udash_preprocessed.parquet"))

	# TODO: To xarray
	# itps_to_xr(df_itps)
	# save
	return df_itps



def main():
	# download_udash(URL)
	# _extract_udash()
	load_udash()


if __name__ == "__main__":
	main()
