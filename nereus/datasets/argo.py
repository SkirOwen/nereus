from __future__ import annotations

import glob
import os
import shutil

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

from tqdm import tqdm

from nereus import logger
from nereus.utils.directories import get_argo_dir


URL = "https://data-argo.ifremer.fr"

RENAME_COL = {
	"PRES": "pres",
	"PSAL": "sal",
	"TEMP": "temp",
	"DOX2_ADJUSTED": "dis_oxy",
}


def extract_argo() -> None:
	argo_dir = get_argo_dir()
	logger.info("Extracting Argo")

	filename = glob.glob("DataSelection*.tar.gz", root_dir=argo_dir)
	print(filename)

	shutil.unpack_archive(
		filename=os.path.join(argo_dir, filename[0]),
		extract_dir=argo_dir,
		format="gztar",
	)


def load_all_argo() -> list:
	all_argo = [
		f for f in os.listdir(get_argo_dir())
		if f.endswith(".nc")
	]

	argos = []
	for f in tqdm(all_argo):
		argos.append(xr.open_dataset(os.path.join(get_argo_dir(), f)))
	return argos


# def interp_argo(argo: pd.DataFrame, dims: list[str], x_inter, base_dim: str, **kwargs) -> pd.DataFrame:
# 	x_inter = np.arange(10, 760, 10)
#
# 	# time = argo.time[i].data
# 	# lat = argo.lat[i].data
# 	# lon = argo.longitude[i].data
#
# 	interp_udash = {
# 		"profile":  np.full(x_inter.shape, argo["profile"].values[0]),  # So everything has the same length
# 		"cruise":   np.full(x_inter.shape, argo["cruise"].values[0]),
# 		"time":     np.full(x_inter.shape, argo.TIME.values[0]),
# 		"lat":      np.full(x_inter.shape, argo.["lat"].values[0]),
# 		"lon":      np.full(x_inter.shape, argo.["lon"].values[0]),
# 		"source":   np.full(x_inter.shape, argo.["source"].values[0]),
# 		base_dim:   x_inter
# 	}
#
# 	for dim in dims:
# 		if dim in argo:
# 			argo[dim] = np.interp(x_inter, argo[base_dim].values, argo[dim].values)
# 		else:
# 			argo[dim] = np.full(x_inter.shape, np.nan)
# 	return pd.DataFrame(argo)


# def clean_argo(argo: xr.Dataset):
# 	argo = argo.drop_dims("POSITION")
# 	argo = argo.drop_vars(
# 		[
# 			"TIME_QC",
# 			"DC_REFERENCE",
# 			"DIRECTION",
# 			"VERTICAL_SAMPLING_SCHEME",
# 			"PRES_QC",
# 			"PRES_ADJUSTED",
# 			"PRES_ADJUSTED_QC",
# 			"TEMP_QC",
# 			"TEMP_ADJUSTED",
# 			"TEMP_ADJUSTED_QC",
# 			"PSAL_QC",
# 			"PSAL_ADJUSTED",
# 			"PSAL_ADJUSTED_QC",
# 		]
# 	)


def process_argo(argo, x_inter=np.arange(10, 760, 10), base_dim="PRES", dims=("TEMP", "PSAL", "DOX2_ADJUSTED")):
	processed = []

	for i in range(argo.TIME.size):
		if (argo.PRES[i].min() > 10.0) | (argo.PRES[i].max() < 750.0) | (len(argo.PRES[i]) <= 2):
			continue

		interp_argo = {
			"time": np.full(x_inter.shape, argo.TIME[i].data),
			"lat": np.full(x_inter.shape, argo.LATITUDE[i].data),
			"lon": np.full(x_inter.shape, argo.LONGITUDE[i].data),
			"profile": np.full(x_inter.shape, f"argo_{argo.id}_{i}"),
			base_dim: x_inter,
		}

		for dim in dims:
			if dim in argo:
				interp_argo[dim] = np.interp(x_inter, argo[base_dim][i].data, argo[dim][i].data)
			else:
				interp_argo[dim] = np.full(x_inter.shape, np.nan)
		processed.append(pd.DataFrame(interp_argo))

	return processed


def argos_to_xr(argos: pd.DataFrame) -> xr.Dataset:
	unique_coords = argos.drop_duplicates("profile").set_index("profile")[["lat", "lon", "time"]]
	argos.set_index(["profile", "pres"], inplace=True)

	ds = xr.Dataset.from_dataframe(argos)
	for coord in ["lat", "lon", "time"]:
		ds = ds.assign_coords({coord: ("profile", unique_coords[coord])})
	return ds


def preload_argo(**kwargs) -> str:
	# check download
	# parse

	save_path = os.path.join(get_argo_dir(), "argos_xr.nc")

	if not os.path.exists(save_path):
		argos = load_all_argo()
		logger.info("Argo Loaded")
		processed_argo = []

		x_inter = np.arange(10, 760, 10)
		base_dim = "PRES"
		dims = ["TEMP", "PSAL", "DOX2_ADJUSTED"]

		# with ProcessPoolExecutor(6) as executor:
		# 	results = executor.map(process_argo, argos)
		# 	for result in tqdm(results, total=len(argos)):
		# 		processed_argo.extend(result)

		for argo in tqdm(argos):
			for i in range(argo.TIME.size):
				if np.all(np.diff(argo.PRES[i]) > 0):   # Making sure no NaNs in the pressure and strictly increasing
					continue
				if (argo.PRES[i].min() > 10.0) | (argo.PRES[i].max() < 750.0) | (len(argo.PRES[i]) <= 2):
					continue

				interp_argo = {
					"time": np.full(x_inter.shape, argo.TIME[i].data),
					"lat": np.full(x_inter.shape, argo.LATITUDE[i].data),
					"lon": np.full(x_inter.shape, argo.LONGITUDE[i].data),
					"profile": np.full(x_inter.shape, f"argo_{argo.id}_{i}"),
					base_dim: x_inter,
				}

				for dim in dims:
					if dim in argo:
						interp_argo[dim] = np.interp(x_inter, argo[base_dim][i].data, argo[dim][i].data)
					else:
						interp_argo[dim] = np.full(x_inter.shape, np.nan)
				processed_argo.append(pd.DataFrame(interp_argo))

		logger.info("Concat")
		argos = pd.concat(processed_argo, ignore_index=True)

		logger.info("Rename")
		argos.rename(columns=RENAME_COL, inplace=True)

		logger.info("Caching")
		argos.to_parquet(os.path.join(get_argo_dir(), "argos_preprocessed.parquet"))

		logger.info("Converting to xarray")
		ds = argos_to_xr(argos)

		logger.info("Saving xr")

		ds.to_netcdf(
			save_path,
			format="NETCDF4",
			engine="h5netcdf",
		)

	return save_path


def main():
	preload_argo()


if __name__ == "__main__":
	main()
