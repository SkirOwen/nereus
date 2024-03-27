from __future__ import annotations

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr


def change_coords(data: xr.Dataset, proj):
	# Create a Plate Carree projection for the original data
	plate_carree_proj = ccrs.PlateCarree()

	# Transform the longitude and latitude coordinates to the North Polar Stereographic projection
	x, y, _ = proj.transform_points(plate_carree_proj, data["lon"], data["lat"]).T

	# Add x and y as coordinates to the existing xarray Dataset
	data.coords['x'] = ('profile', x)
	data.coords['y'] = ('profile', y)

	return data


def grid_bin(data: xr.Dataset, x_step: float, y_step: float, scale: float = 1):
	# lat_step = 200_000
	# lon_step = 200_000
	df_latlon = data[['y', 'x']].to_dataframe()
	lon_bins = pd.cut(df_latlon['y'], bins=np.arange(min(df_latlon['y']), max(df_latlon['y']), y_step / scale))
	lat_bins = pd.cut(df_latlon['x'], bins=np.arange(min(df_latlon['x']), max(df_latlon['x']), x_step / scale))
	latlon_groups = df_latlon.groupby([lon_bins, lat_bins], observed=False)
	return latlon_groups


def main():
	pass


if __name__ == "__main__":
	main()
