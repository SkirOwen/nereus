from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.path as mpath

import numpy as np
import pandas as pd
import seaborn as sns
import cmocean as cm
import cartopy.crs as ccrs

from nereus import logger


def map_itps(itps_lat_lon: list):
	lat, lon = itps_lat_lon

	fig = plt.figure(figsize=(10, 10), dpi=300)
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
	ax.set_extent([-180, 180, 90, 55], ccrs.PlateCarree())
	theta = np.linspace(0, 2 * np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.stock_img()
	ax.set_boundary(circle, transform=ax.transAxes)

	ax.scatter(lon, lat, transform=ccrs.PlateCarree())
	plt.show()


def time_hist(metadatas) -> None:
	years = pd.to_datetime(metadatas.time).dt.year.values
	months = pd.to_datetime(metadatas.time).dt.month.values

	fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
	bins_month = np.arange(1, 14)  # set bin edges
	axs[0].hist(months, bins=bins_month)
	axs[0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1])
	axs[0].set_title('Month')
	bins_year = np.arange(2004, 2022)  # set bin edges
	axs[1].hist(years, bins=bins_year)
	axs[1].set_xticks(bins_year)
	axs[1].set_xticklabels(bins_year, rotation=90)  # rotate xticklabels by 90 degrees
	axs[1].set_title('Year')
	plt.show()


def main():
	pass


if __name__ == "__main__":
	main()
