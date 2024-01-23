from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.path as mpath

import numpy as np
import pandas as pd
import seaborn as sns
import cmocean as cm
import cartopy.crs as ccrs

from rich.console import Console

from nereus import logger
from nereus.utils.directories import get_plot_dir

from nereus.datasets import load_udash, load_itp


def get_arctic_map():
	fig = plt.figure(figsize=(10, 10), dpi=300)
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
	ax.set_extent([-180, 180, 90, 55], ccrs.PlateCarree())
	theta = np.linspace(0, 2 * np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.coastlines()
	ax.set_boundary(circle, transform=ax.transAxes)
	return fig, ax


def map_itps(itps_metadata):
	dt_itps = itps_metadata.drop_duplicates(["longitude(E+)", "latitude(N+)", "time"]).sort_index()

	fig, ax = get_arctic_map()

	logger.info("Scatter plot, takes some time")
	with Console().status("Loading") as st:
		g = sns.scatterplot(
			data=dt_itps,
			x="longitude(E+)",
			y="latitude(N+)",
			hue="time",
			s=1,
			ax=ax,
			transform=ccrs.PlateCarree(),
			markers="h",
		)

	logger.info("Removing legend")
	g.legend_.remove()
	plt.legend([], [], frameon=False)

	logger.info("Saving to file")
	plt.savefig(os.path.join(get_plot_dir(), "itps_map.png"))

	plt.show()


def map_udash(udash):
	logger.info("Removing duplicate location and time")
	# TODO: could have the number with size?
	dt_udash = udash.drop_duplicates(["Latitude_[deg]", "Longitude_[deg]", "yyyy-mm-ddThh:mm"]).sort_index()

	fig, ax = get_arctic_map()

	logger.info("Scatter plot, takes some time")
	with Console().status("Loading") as st:
		g = sns.scatterplot(
			data=dt_udash,
			x="Longitude_[deg]",
			y="Latitude_[deg]",
			hue="yyyy-mm-ddThh:mm",
			s=1,
			ax=ax,
			transform=ccrs.PlateCarree(),
			markers="h",
		)

	logger.info("Removing legend")
	g.legend_.remove()
	plt.legend([], [], frameon=False)

	logger.info("Saving to file")
	plt.savefig(os.path.join(get_plot_dir(), "udash_map.png"))
	plt.show()


def udash_depth_hist(udash):
	logger.info("Plotting depth histogram")
	sns.displot(udash, x="Depth_[m]")
	plt.savefig(os.path.join(get_plot_dir(), "udash_depth_hist.png"), dpi=1000)
	plt.show()


def udash_time_hist(udash):
	logger.info("Plotting date hist")
	sns.displot(udash, x="yyyy-mm-ddThh:mm")
	plt.savefig(os.path.join(get_plot_dir(), "udash_time_hist.png"), dpi=1000)
	plt.show()


def udash_months_hist(udash):
	logger.info("Plotting month hist")
	udash_months = udash["yyyy-mm-ddThh:mm"].dt.month
	sns.histplot(udash_months, discrete=True)
	plt.savefig(os.path.join(get_plot_dir(), "udash_months_hist.png"), dpi=1000)
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
	udash = load_udash()
	map_udash(udash)
	# udash_depth_hist(udash)
	# udash_months_hist(udash)
	# udash_time_hist(udash)
	# fig, ax = get_arctic_map()
	# plt.show()
	itps, metadata = load_itp()
	map_itps(metadata)
	# time_hist(metadata)


if __name__ == "__main__":
	main()
