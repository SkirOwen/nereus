from __future__ import annotations

import gc
import glob
import os
import shutil

from datetime import date, datetime
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xarray as xr

from tqdm import tqdm

from nereus import logger
from nereus.utils.directories import get_udash_dir, get_udash_extracted_dir
from nereus.utils.downloader import downloader


URL = "https://hs.pangaea.de/Projects/UDASH/UDASH.zip"

UDASH_COLUMN_TYPE = {
	"Prof_no": int,
	"Cruise": str,
	"Station": str,
	"Platform": str,
	"Type": str,
	# 'yyyy-mm-ddThh:mm': datetime,
	"Longitude_[deg]": float,
	"Latitude_[deg]": float,
	"Pressure_[dbar]": float,
	"Depth_[m]": float,
	"QF_Depth_[m]": int,
	"Temp_[C]": float,
	"QF_Temp_[C]": int,
	"Salinity_[psu]": float,
	"QF_Salinity_[psu]": int,
	"Source": str,
	"DOI": str,
	"WOD-Cruise-ID": str,
	"WOD-Cast-ID": str,
}

rename_col = {
	"Prof_no": "profile",
	"Cruise": "cruise",
	"Station": "station",
	"Platform": "platform",
	"Type": "type",
	"yyyy-mm-ddThh:mm": "time",
	"Longitude_[deg]": "lon",
	"Latitude_[deg]": "lat",
	"Pressure_[dbar]": "pres",
	"Depth_[m]": "depth",
	# 'QF_Depth_[m]': int,
	"Temp_[C]": "temp",
	# 'QF_Temp_[C]': int,
	"Salinity_[psu]": "sal",
	# 'QF_Salinity_[psu]': int,
	"Source": "source",
	# 'DOI': str,
	# 'WOD-Cruise-ID': str,
	# 'WOD-Cast-ID': str,
}


def download_udash(url: str, override: bool = False) -> None:
	udash_dir = get_udash_dir()
	downloader(url, udash_dir, override=override)


def _extract_udash() -> None:
	udash_dir = get_udash_dir()
	logger.info("Extracting UDASH.")

	shutil.unpack_archive(
		filename=os.path.join(udash_dir, "UDASH.zip"),
		extract_dir=udash_dir,
		format="zip",
	)


def _udash_fileparser(
		filepath: str, filtering: bool = True, remove_argo: bool = True, remove_itp: bool = True
) -> pd.DataFrame:
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

	for line in lines[1:]:
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
	# logger.info("Clean")
	data = clean_df_udash(data)

	# logger.info("argo")
	if remove_argo:
		data.query('Source != "argo"', inplace=True)
	# logger.info("itp")
	if remove_itp:
		data.query('not Cruise.str.contains("itp")', inplace=True)

	# logger.info("filter")
	if filtering:
		# Use the filter method to apply the conditions to each group
		data = data.groupby("Prof_no").filter(
			partial(filter_groups, dim="Pressure_[dbar]", low=10.0, high=750.0, min_nobs=2)
		)
	return data


def clean_df_udash(data, col_to_drop: None | list[str] = None) -> pd.DataFrame:
	col_to_drop = [
		"Station",
		"Platform",
		"Type",
		"Depth_[m]",
		"QF_Depth_[m]",
		"QF_Temp_[C]",
		"QF_Salinity_[psu]",
		"DOI",
		"WOD-Cruise-ID",
		"WOD-Cast-ID",
	]
	data.drop(
		col_to_drop,
		axis=1,
		inplace=True,
	)

	for c in col_to_drop:
		if c in UDASH_COLUMN_TYPE:
			UDASH_COLUMN_TYPE.pop(c)

	data = data.astype(UDASH_COLUMN_TYPE)
	data.replace({-999: np.nan, "-999": np.nan}, inplace=True)
	data["yyyy-mm-ddThh:mm"] = pd.to_datetime(data["yyyy-mm-ddThh:mm"], errors="coerce")
	# data.rename(columns=rename_col, inplace=True)
	# data["time"] = pd.to_datetime(data["time"], errors="coerce")    # This puts NaT for wrong time
	# print(os.path.basename(filepath), data.time.isnull().sum())
	# data = data[data.time.notnull()]                                # This filters the NaT and drop them

	return data


def parse_all_udash(files_nbr: None | int = None, **kwargs) -> pd.DataFrame:
	udash_extracted_dir = get_udash_extracted_dir()
	files = glob.glob(os.path.join(udash_extracted_dir, "ArcticOcean_*.txt"))

	if files_nbr is not None:
		files = files[:files_nbr]

	logger.info(f"{len(files)} udash files to parse.")

	udash = []

	with Pool(6) as pool:
		for data in tqdm(
			pool.imap(partial(_udash_fileparser, **kwargs), files), total=len(files), desc="Parsing udash"
		):
			udash.append(data)
			gc.collect()

	udash = pd.concat(udash, ignore_index=True)

	return udash


def filter_groups(group, dim, low, high, min_nobs):
	mask = group[dim].max() >= high and group[dim].min() <= low and len(group[dim]) > min_nobs
	return mask


def interp_udash(udash: pd.DataFrame, dims: list[str], x_inter, base_dim: str) -> pd.DataFrame:
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
	interp_udash_dict = {
		"profile":  np.full(x_inter.shape, udash["profile"].values[0]),  # So everything has the same length
		"cruise":   np.full(x_inter.shape, udash["cruise"].values[0]),
		"time":     np.full(x_inter.shape, udash["time"].values[0]),
		"lat":      np.full(x_inter.shape, udash["lat"].values[0]),
		"lon":      np.full(x_inter.shape, udash["lon"].values[0]),
		"source":   np.full(x_inter.shape, udash["source"].values[0]),
		base_dim:   x_inter,
	}

	for dim in dims:
		if dim in udash:
			interp_udash_dict[dim] = np.interp(x_inter, udash[base_dim].values, udash[dim].values)
		else:
			interp_udash_dict[dim] = np.full(x_inter.shape, np.nan)
	return pd.DataFrame(interp_udash_dict)


def udash_to_xr(udash: pd.DataFrame) -> xr.Dataset:
	unique_coords = udash.drop_duplicates("profile").set_index("profile")[["lat", "lon", "time", "cruise", "source"]]
	udash.set_index(["profile", "pres"], inplace=True)

	ds = xr.Dataset.from_dataframe(udash)
	for coord in ["lat", "lon", "time", "cruise", "source"]:
		ds = ds.assign_coords({coord: ("profile", unique_coords[coord])})
	return ds


def preload_udash(**kwargs) -> str:
	# check download
	# parse
	save_path = os.path.join(get_udash_dir(), "udash_xr.nc")
	if not os.path.exists(save_path):
		if not os.path.exists(os.path.join(get_udash_dir(), "udash_preprocessed.parquet")):
			udash = parse_all_udash()
			udash.rename(columns=rename_col, inplace=True)
			logger.info("Parsed")
			processed_udash = []

			for i, u in tqdm(udash.groupby("profile")):
				new_u = interp_udash(u, **kwargs)
				processed_udash.append(new_u)

			logger.info("Concat")
			udash = pd.concat(processed_udash, ignore_index=True)

			udash = udash[udash.time.notnull()]
			udash.reset_index(inplace=True)

			logger.info("Caching")
			udash.to_parquet(os.path.join(get_udash_dir(), "udash_preprocessed.parquet"))
		else:
			udash = pd.read_parquet(os.path.join(get_udash_dir(), "udash_preprocessed.parquet"))

		logger.info("Converting to xarray")
		ds = udash_to_xr(udash)

		logger.info("Saving xr")

		ds.to_netcdf(
			save_path,
			format="NETCDF4",
			engine="h5netcdf",
		)
	return save_path


def main():
	# preload_udash(dims=["temp", "sal", "depth"], x_inter=None, base_dim="pres")

	udash = parse_all_udash(filtering=False, remove_argo=False, remove_itp=False)
	print(len(udash))

	# download_udash(URL)
	# _extract_udash()
	# load_udash()


if __name__ == "__main__":
	main()
