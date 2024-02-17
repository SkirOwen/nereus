from __future__ import annotations

import os

import pandas as pd
import numpy as np
import xarray as xr

from nereus.datasets.itp import select_range


def df_merge_quick(itps: tuple, udash: pd.DataFrame, argo: list[xr.Dataset]) -> pd.DataFrame:
	pass


def format_merged(ds, **kwargs) -> xr.Dataset:
	return ds


def regen_all_datasets(**kwargs) -> xr.Dataset:
	# This would return the path of the cache file, to preserve memory
	# And if I want or system can, this could be async/parallel
	itps: str = preload_itp(**kwargs)
	udash: str = preload_udash(**kwargs)
	argos: str = preload_argos(**kwargs)

	ds = xr.open_mfdataset([itps, udash, argos])

	ds = format_merged(ds)

	if kwargs["save"]:
		cache_path = ""
		ds.to_netcdf(cache_path)

	return ds


def load(**kwargs) -> xr.Dataset:
	cache_file = ""
	if os.path.exists(cache_file):
		ds = xr.open_dataset(cache_file)
	else:
		ds = regen_all_datasets(**kwargs)
	return ds


def main():
	pass

if __name__ == "__main__":
	main()
