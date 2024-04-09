from __future__ import annotations

import os

import xarray as xr

from nereus.datasets.itp import preload_itp
from nereus.datasets.argo import preload_argo
from nereus.datasets.udash import preload_udash


def format_merged(ds, **kwargs) -> xr.Dataset:
	return ds


def regen_all_datasets(**kwargs) -> xr.Dataset:
	# This would return the path of the cache file, to preserve memory
	# And if I want or system can, this could be async/parallel
	itps_file: str = preload_itp(**kwargs)
	udash_file: str = preload_udash(**kwargs)
	argo_file: str = preload_argo(**kwargs)

	ds = xr.open_mfdataset([itps_file, udash_file, argo_file], chunks="auto", parallel=True)

	ds = format_merged(ds)

	# if kwargs["save"]:
	# 	cache_path = ""
	# 	ds.to_netcdf(cache_path)
	return ds


def load_data(**kwargs) -> xr.Dataset:
	cache_file = ""
	if os.path.exists(cache_file):
		ds = xr.open_dataset(cache_file)
	else:
		ds = regen_all_datasets(**kwargs)
	return ds


def main():
	load_data()


if __name__ == "__main__":
	main()
