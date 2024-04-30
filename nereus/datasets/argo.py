from __future__ import annotations

import glob
import os
import shutil

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from rich.progress import Progress, TaskID
from tqdm.auto import tqdm

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
	for f in tqdm(all_argo, desc="Opening Argos"):
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


def process_argo(
		argo: xr.Dataset,
		x_inter: np.ndarray = np.arange(10, 760, 10),
		base_dim: str = "PRES",
		dims: Iterable[str] = ("TEMP", "PSAL", "DOX2_ADJUSTED"),
		task_tot: int | None = None,
		task_num: int | None = None,
	) -> list:
	""""""
	processed_argo = []
	with tqdm(total=argo.TIME.size, desc=f"(Task {task_num} / {task_tot}) Process date", position=1, leave=False) as pbar:
		for i in range(argo.TIME.size):
			if np.all(np.diff(argo.PRES[i]) > 0):  # Making sure no NaNs in the pressure and strictly increasing
				continue
			if (argo.PRES[i].min() > 10.0) | (argo.PRES[i].max() < 750.0) | (np.count_nonzero(~np.isnan(argo.PRES[i])) <= 2):
				continue
			if any([(np.count_nonzero(~np.isnan(argo[dim][i])) <= 2) for dim in dims if dim != "DOX2_ADJUSTED"]):
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
			pbar.update(1)
		pbar.close()

	return processed_argo


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
	parallel = True

	save_path = os.path.join(get_argo_dir(), "argos_xr.nc")

	if not os.path.exists(save_path):
		argos = load_all_argo()
		logger.info("Argo Loaded")
		processed_argos = []

		x_inter = np.arange(10, 760, 10)
		base_dim = "PRES"
		dims = ["TEMP", "PSAL", "DOX2_ADJUSTED"]

		if parallel:
			with ProcessPoolExecutor(6) as executor:
				partial_process = partial(
					process_argo,
					x_inter=x_inter,
					dims=dims,
					base_dim=base_dim,
					task_tot=len(argos),
				)
				futures = [executor.submit(partial_process, argo, task_num=i) for i, argo in enumerate(argos)]

				for future in tqdm(futures, desc="Processing argos", position=0):
					processed_argos.extend(future.result())

		else:
			for argo in tqdm(argos):
				processed_argo = process_argo(argo, x_inter, base_dim, dims)
				processed_argos.extend(processed_argo)

		logger.info("Concat")
		argos = pd.concat(processed_argos, ignore_index=True)

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
