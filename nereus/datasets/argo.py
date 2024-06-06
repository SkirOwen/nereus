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
	all_argo = glob.glob(
		os.path.join(get_argo_dir(), "GL_*.nc")
	)

	argos = []
	for f in tqdm(all_argo, desc="Opening Argos"):
		argos.append(xr.open_dataset(os.path.join(get_argo_dir(), f)))
	return argos


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
	min_pres = min(x_inter)
	max_pres = max(x_inter)

	for i in tqdm(range(argo.TIME.size), desc=f"(Task {task_num} / {task_tot}) Process date", position=task_num % 6 + 1, leave=False):
		if not np.all([dim in argo.data_vars for dim in dims if dim != "DOX2_ADJUSTED"]):
			# TODO: Should this be only for TEMP and SAL?
			continue
		if not np.all(np.diff(argo.PRES[i]) > 0):  # Making sure no NaNs in the pressure and strictly increasing
			continue
		if (argo.PRES[i].min() > min_pres) | (argo.PRES[i].max() < max_pres) | (np.count_nonzero(~np.isnan(argo.PRES[i])) <= 2):
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

	return processed_argo


def argos_to_xr(argos: pd.DataFrame) -> xr.Dataset:
	unique_coords = argos.drop_duplicates("profile").set_index("profile")[["lat", "lon", "time"]]
	argos.set_index(["profile", "pres"], inplace=True)

	ds = xr.Dataset.from_dataframe(argos)
	for coord in ["lat", "lon", "time"]:
		ds = ds.assign_coords({coord: ("profile", unique_coords[coord])})
	return ds


def preload_argo(parallel: bool = True, regen: bool = False, low_filter: int = 10, high_filter: int = 750, **kwargs) -> str:
	save_path = os.path.join(get_argo_dir(), "argos_xr.nc")

	if not os.path.exists(save_path) or regen:
		cache_path = os.path.join(get_argo_dir(), "argos_preprocessed.parquet")
		if not os.path.exists(cache_path) or regen:
			argos = load_all_argo()
			logger.info("Argo Loaded")
			processed_argos = []

			x_inter = np.arange(low_filter, high_filter + 10, 10)
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
						result = future.result()
						# if result is not None:
						processed_argos.extend(result)

			else:
				for i, argo in enumerate(tqdm(argos)):
					processed_argo = process_argo(argo, x_inter, base_dim, dims, task_tot=len(argo), task_num=i)
					if processed_argo is not None:
						processed_argos.extend(processed_argo)

			logger.info("Concat")
			argos = pd.concat(processed_argos, ignore_index=True)

			logger.info("Rename")
			argos.rename(columns=RENAME_COL, inplace=True)

			logger.info("Caching")
			argos.to_parquet(cache_path)
		else:
			argos = pd.read_parquet(cache_path)

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
	preload_argo(regen=True)


if __name__ == "__main__":
	main()
