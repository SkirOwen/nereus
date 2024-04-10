from __future__ import annotations

import numpy as np
import os
import xarray as xr
import cartopy.crs as ccrs

from nereus import logger
from nereus.processing.coordinates_utils import grid_bin_coord
from nereus.utils.directories import get_data_dir
from nereus.datasets.merged import load_data


def prepare_training_data(
		lat_step: float,
		lon_step: float,
		quantile4T: float,
		ds_cleaned: xr.Dataset,
		ratio_monthly_sampled: float = 1,
		proj=ccrs.NorthPolarStereo(),
) -> xr.Dataset:
	"""
	Prepare the training data by spatially and temporally sample the cleaned data.

	Args:
		lat_step (float): The latitude step size.
		lon_step (float): The longitude step size.
		quantile4T (float): The quantile value for threshold calculation.
		ds_cleaned (xr.Dataset): The cleaned dataset.
		ratio_monthly_sampled (float, optional): The ratio for monthly sampling. Defaults to 1.

	Returns:
		xr.Dataset: The prepared training data.
	"""

	logger.info("Calculating threshold")
	threshold = int(cal_threshold(ds_cleaned, lat_step, lon_step, quantile=quantile4T, proj=proj))

	logger.info("Drop profiles")
	ds_cleaned_spatially_sampled = drop_profiles(ds_cleaned, lat_step, lon_step, threshold, proj=proj)

	logger.info("Random sample by month")
	ds_cleaned_monthly_sampled = ramdon_sample_by_month(
		ds_cleaned_spatially_sampled,
		ratio_monthly_sampled,
		random_seed=0
	)
	ds_cleaned_monthly_sampled['lat_step'] = lat_step
	ds_cleaned_monthly_sampled['lon_step'] = lon_step
	ds_cleaned_monthly_sampled['quantile4T'] = quantile4T
	print(f'threshold={threshold}, training dataset size={len(ds_cleaned_monthly_sampled.profile)}')
	return ds_cleaned_monthly_sampled


def cal_threshold(
		ds: xr.Dataset,
		lat_step: float,
		lon_step: float,
		quantile: float,
		proj: ccrs.CRS,
) -> int:
	"""
	Calculate the threshold based on the difference between adjacent elements in the sorted latitude-longitude bins.

	Args:
		ds (xr.Dataset): The dataset.
		lat_step (float): The latitude step size for binning.
		lon_step (float): The longitude step size for binning.
		quantile (float): The quantile value for threshold calculation.

	Returns:
		int: The calculated threshold.
	"""

	latlon_groups = grid_bin_coord(ds, proj, x_step=lat_step, y_step=lon_step)

	g_len = [len(group[1]) for group in latlon_groups]

	g_len_sorted = sorted(g_len)
	transition_point = None

	g_len_diff = np.diff(g_len_sorted)

	threshold = np.quantile(g_len_diff, quantile)  # set a threshold for the difference between adjacent elements

	for i in range(1, len(g_len_sorted)):
		if g_len_sorted[i] - g_len_sorted[i - 1] > threshold:
			transition_point = i - 1
			break

	return g_len_sorted[transition_point]


def drop_profiles(
		ds: xr.Dataset,
		lat_step: float,
		lon_step: float,
		threshold: int,
		proj: ccrs.CRS,
) -> xr.Dataset:
	"""
	Drop the profiles in the bins with sample size larger than the threshold.

	Args:
		ds (xr.Dataset): The dataset.
		lat_step (float): The latitude step size for binning.
		lon_step (float): The longitude step size for binning.
		threshold (int): The threshold for dropping profiles.

	Returns:
		xr.Dataset: The dataset with profiles dropped.
	"""
	latlon_groups = grid_bin_coord(ds, proj, x_step=lat_step, y_step=lon_step)

	latlon_groups_list = []

	for group in latlon_groups:
		if len(group[1]) > threshold:
			group_drop = group[1].sample(n=len(group[1]) - threshold, random_state=0)
			latlon_groups_list.append(group_drop)

	ds_spatially_sampled = ds.copy()

	nprof_dropped = []

	for group in latlon_groups_list:
		temp_index = group.index.values
		nprof_dropped.append(temp_index.tolist())

	nprof_dropped = [item for sublist in nprof_dropped for item in sublist]

	mask = ~ds_spatially_sampled["profile"].isin(nprof_dropped)
	ds_spatially_sampled = ds_spatially_sampled.where(mask, drop=True)

	# ds_spatially_sampled = ds_spatially_sampled.drop_sel(profile=nprof_dropped)

	return ds_spatially_sampled


def ramdon_sample_by_month(
		ds: xr.Dataset,
		ratio: float,
		random_seed: int = 0
) -> xr.Dataset:
	"""
	Sample profiles by month based on a given ratio.

	Args:
		ds (xr.Dataset): The dataset.
		ratio (float): The ratio of profiles to sample (between 0 and 1).
		random_seed (int, optional): The random seed for reproducibility. Defaults to 0.

	Returns:
		xr.Dataset: The dataset with sampled profiles.
	"""
	if ratio < 0 or ratio > 1:
		raise ValueError("Ratio must be between 0 and 1.")

	if 'time' not in ds:
		raise ValueError("Input dataset must have a 'time' variable.")

	if 'profile' not in ds.dims:
		raise ValueError("Input dataset must have a 'profile' dimension.")

	ds_monthly = ds.groupby('time.month')
	months_with_data = [month for month in ds_monthly.groups.keys() if len(ds_monthly[month]) > 0]

	if not months_with_data:
		raise ValueError("Input dataset does not contain any data.")

	smallest_month_size = np.min([ds_monthly[month].profile.size for month in months_with_data])
	selected_size_monthly = ds.profile.size * ratio / 12

	if smallest_month_size < selected_size_monthly:
		selected_size_monthly = smallest_month_size
		print(f"Ratio is too large, selected_size_monthly is reset to the smallest_month_size: {smallest_month_size}")

	ds_selected = xr.concat(
		[ds_monthly[month].isel(
			profile=np.random.default_rng(random_seed).choice(
				ds_monthly[month].profile.size,
				int(selected_size_monthly), replace=False
			)
		) for month in months_with_data],
		dim='profile')

	ds_selected['profile'] = np.arange(ds_selected.sizes['profile'])

	return ds_selected


def main():
	ds = load_data().load()
	x_step = 290_000
	y_step = 300_000
	train_ds = prepare_training_data(lat_step=x_step, lon_step=y_step, quantile4T=0.95, ds_cleaned=ds)

	logger.info("save")
	train_ds.to_netcdf(
		os.path.join(get_data_dir(), "train_ds.nc"),
		format="NETCDF4",
		engine="h5netcdf",
	)


if __name__ == "__main__":
	main()
