from __future__ import annotations

import os

import xarray as xr

from nereus.datasets.argo import preload_argo
from nereus.datasets.itp import preload_itp
from nereus.datasets.udash import preload_udash
from nereus.utils.file_ops import create_cache_filename
from nereus.utils.directories import get_data_dir


def format_merged(ds, **kwargs) -> xr.Dataset:
	return ds


def regen_all_datasets(**kwargs) -> xr.Dataset:
	# This would return the path of the cache file, to preserve memory
	# And if I want or system can, this could be async/parallel
	itps_file: str = preload_itp(**kwargs)
	udash_file: str = preload_udash(**kwargs)
	argo_file: str = preload_argo(**kwargs)

	ds = xr.open_mfdataset([itps_file, udash_file, argo_file], chunks="auto", parallel=True)

	ds = format_merged(ds, **kwargs)

	if kwargs["save"]:
	# 	cache_path = create_cache_filename(name="merged", **kwargs)
		cache_path = os.path.join(get_data_dir(), "test.nc")
		ds.to_netcdf(
			cache_path,
			format="NETCDF4",
			engine="h5netcdf",
		)
	return ds


def load_data(**kwargs) -> xr.Dataset:
	cache_file = create_cache_filename(name="merged", **kwargs)
	if os.path.exists(cache_file):
		ds = xr.open_dataset(cache_file)
	else:
		ds = regen_all_datasets(save=False, **kwargs)
	return ds


def main():
	load_data(save=True)


if __name__ == "__main__":
	main()
