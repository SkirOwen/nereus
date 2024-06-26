from __future__ import annotations

import os

import cartopy.crs as ccrs
import cmocean as cm
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rich.console import Console

from nereus import logger
from nereus.utils.directories import get_plot_dir


def get_arctic_map(ax: Axes | None = None, labels: bool = False) -> tuple[Figure, Axes] | Axes:
	if ax is None:
		fig = plt.figure(figsize=(10, 10), dpi=300)
		ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
		ax_created = True
	else:
		ax_created = False
	ax.set_extent([-180, 180, 90, 55], ccrs.PlateCarree())
	theta = np.linspace(0, 2 * np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.coastlines()
	ax.set_boundary(circle, transform=ax.transAxes)

	# Adjust longitude and latitude labels
	if labels:
		gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
		gl.xlocator = mticker.FixedLocator(np.concatenate([np.arange(-180, 180, 20), np.arange(-180, 180, 20)]))
		gl.xformatter = LONGITUDE_FORMATTER
		gl.xlabel_style = {"size": 11, "color": "k", "rotation": 0}
		gl.yformatter = LATITUDE_FORMATTER
		gl.ylocator = mticker.FixedLocator(np.arange(65, 90, 5), 200)
		gl.ylabel_style = {"size": 11, "color": "k", "rotation": 0}

	if ax_created:
		return fig, ax
	else:
		return ax


def map_arctic_value(df, name=None, **snskwargs):
	fig, ax = get_arctic_map()

	logger.info("Scatter plot, takes some time")
	with Console().status("Loading") as st:
		sns.scatterplot(
			data=df,
			x="lon",
			y="lat",
			ax=ax,
			transform=ccrs.PlateCarree(),
			markers="h",
			**snskwargs,
		)

	ax.legend(markerscale=2)
	fig.tight_layout()

	logger.info("Saving")
	plt.savefig(os.path.join(get_plot_dir(), f"map_{name}.png"))
	logger.info("Done")
	logger.info("Showing")
	plt.show()


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
	plt.legend([], [], frameon=False, markersize=4)

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
	axs[0].set_title("Month")
	bins_year = np.arange(2004, 2022)  # set bin edges
	axs[1].hist(years, bins=bins_year)
	axs[1].set_xticks(bins_year)
	axs[1].set_xticklabels(bins_year, rotation=90)  # rotate xticklabels by 90 degrees
	axs[1].set_title("Year")
	plt.show()


def all_spatial_old(metadata, udash, argos):
	itp_lat = metadata["latitude(N+)"].values
	udash_lat = udash.lat.data
	argo_lat = np.concatenate([a.LATITUDE.data for a in argos])

	itp_lon = metadata["longitude(E+)"].values
	udash_lon = udash.lon.data
	argo_lon = np.concatenate([a.LONGITUDE.data for a in argos])

	df_itp = pd.DataFrame({"lat": itp_lat, "lon": itp_lon, "source": "itp"})
	df_udash = pd.DataFrame({"lat": udash_lat, "lon": udash_lon, "source": "udash"})
	df_argo = pd.DataFrame({"lat": argo_lat, "lon": argo_lon, "source": "argo"})
	# Concatenate all DataFrames
	combined_df = pd.concat([df_itp, df_udash, df_argo])
	combined_df["source"] = combined_df["source"].astype(str)
	combined_df_droped = combined_df.drop_duplicates(["lat", "lon", "source"])

	fig, ax = get_arctic_map()

	sns.scatterplot(
		data=combined_df_droped,
		x="lon",
		y="lat",
		hue="source",
		s=1,
		ax=ax,
		transform=ccrs.PlateCarree(),
		markers="h",
	)

	plt.savefig(os.path.join(get_plot_dir(), "all_spatial.png"), dpi=1000)
	plt.show()


def all_spatial_xr(itps: xr.Dataset, udash: xr.Dataset, argos: xr.Dataset):
	itp_lat = itps.lat.data
	udash_lat = udash.lat.data
	argo_lat = argos.lat.data

	itp_lon = itps.lon.data
	udash_lon = udash.lon.data
	argo_lon = argos.lon.data

	df_itp = pd.DataFrame({"lat": itp_lat, "lon": itp_lon, "source": "itp"})
	df_udash = pd.DataFrame({"lat": udash_lat, "lon": udash_lon, "source": "udash"})
	df_argo = pd.DataFrame({"lat": argo_lat, "lon": argo_lon, "source": "argo"})
	# Concatenate all DataFrames
	combined_df = pd.concat([df_itp, df_udash, df_argo])
	combined_df["source"] = combined_df["source"].astype(str)
	combined_df_droped = combined_df.drop_duplicates(["lat", "lon", "source"])

	fig, ax = get_arctic_map()

	sns.scatterplot(
		data=combined_df_droped,
		x="lon",
		y="lat",
		hue="source",
		s=3,
		ax=ax,
		transform=ccrs.PlateCarree(),
		markers="h",
	)
	plt.legend(title="Sources", markerscale=4)
	plt.tight_layout()
	plt.savefig(os.path.join(get_plot_dir(), "all_spatial.png"), dpi=1000)
	plt.show()


def all_time(metadata, udash: xr.Dataset, argos):
	itp_time = pd.to_datetime(metadata.time.values).values
	udash_time = udash.time.data
	argo_time = np.concatenate([a.TIME.data for a in argos])

	df_itp = pd.DataFrame({"time": itp_time, "source": "itp"})
	df_udash = pd.DataFrame({"time": udash_time, "source": "udash"})
	df_argo = pd.DataFrame({"time": argo_time, "source": "argo"})
	# Concatenate all DataFrames
	combined_df = pd.concat([df_itp, df_udash, df_argo]).reset_index(drop=True)
	combined_df_drop = combined_df.drop_duplicates(["time", "source"]).reset_index(drop=True)

	sns.displot(data=combined_df_drop, x="time", hue="source", discrete=True, kde=False, alpha=0.5)
	plt.ylim(top=200)

	plt.savefig(os.path.join(get_plot_dir(), "all_time.png"), dpi=1000)
	plt.show()


def plot_press(itp, udash, argos):
	itp_press = itp["pressure(dbar)"].values
	udash_press = udash["Pressure_[dbar]"].data
	argo_press = np.concatenate([a.PRES.data.flatten() for a in argos if "PRES" in a])

	df_itp = pd.DataFrame({"press": itp_press, "source": "itp"})
	df_udash = pd.DataFrame({"press": udash_press, "source": "udash"})
	df_argo = pd.DataFrame({"press": argo_press, "source": "argo"})
	# Concatenate all DataFrames
	combined_df = pd.concat([df_itp, df_udash, df_argo]).replace(-999, np.nan).reset_index(drop=True)
	# combined_df = combined_df.drop_duplicates(["press", "source"]).reset_index(drop=True)

	sns.histplot(data=combined_df, x="press", hue="source", discrete=True, kde=False, alpha=0.5)

	plt.savefig(os.path.join(get_plot_dir(), "udash_press.png"), dpi=1000)
	plt.show()


def plot_range_press(itp, udash, argos):
	itp_press_max = itp.groupby("file")["pressure(dbar)"].max()
	itp_press_min = itp.groupby("file")["pressure(dbar)"].min()

	udash = udash.set_coords("Prof_no")
	udash_press_max = udash["Pressure_[dbar]"].groupby("Prof_no").max().data
	udash_press_min = udash["Pressure_[dbar]"].groupby("Prof_no").min().data

	argo_press_max = np.concatenate([a.PRES.max(skipna=True).data.flatten() for a in argos if "PRES" in a])
	argo_press_min = np.concatenate([a.PRES.min(skipna=True).data.flatten() for a in argos if "PRES" in a])

	df_itp = pd.DataFrame({"press_max": itp_press_max, "press_min": itp_press_min, "source": "itp"})
	df_udash = pd.DataFrame({"press_max": udash_press_max, "press_min": udash_press_min, "source": "udash"})
	df_argo = pd.DataFrame({"press_max": argo_press_max, "press_min": argo_press_min, "source": "argo"})
	# Concatenate all DataFrames
	combined_df = pd.concat([df_itp, df_udash, df_argo]).replace(-999, np.nan).reset_index(drop=True)
	# combined_df = combined_df.drop_duplicates(["press", "source"]).reset_index(drop=True)

	sns.histplot(data=combined_df, x="press_max", hue="source", binwidth=100, kde=False, alpha=0.5)
	plt.yscale("symlog")
	plt.title("Max pressure per profile")
	plt.savefig(os.path.join(get_plot_dir(), "press_max.png"), dpi=1000)
	plt.show()

	sns.histplot(data=combined_df, x="press_min", hue="source", binwidth=5, kde=False, alpha=0.5)
	plt.xlim(-10, 6000)
	plt.yscale("symlog")
	plt.title("Min pressure per profile")
	plt.savefig(os.path.join(get_plot_dir(), "press_min.png"), dpi=1000)
	plt.show()


def t_s(itp, udash, argos):
	itp_temp = itp["temperature(C)"].values
	udash_temp = udash["Temp[C]"].data
	argo_temp = np.concatenate([a.TEMP.data.flatten() for a in argos if "TEMP" in a and "PSAL" in a])

	itp_sal = itp["salinity"].values
	udash_sal = udash["Salinity_[psu]"].data
	argo_sal = np.concatenate([a.PSAL.data.flatten() for a in argos if "TEMP" in a and "PSAL" in a])

	df_itp = pd.DataFrame({"temp": itp_temp, "sal": itp_sal, "source": "itp"})
	df_udash = pd.DataFrame({"temp": udash_temp, "sal": udash_sal, "source": "udash"})
	df_argo = pd.DataFrame({"temp": argo_temp, "sal": argo_sal, "source": "argo"})

	ts_combined_df = pd.concat([df_itp, df_udash, df_argo]).replace(-999, np.nan).reset_index(drop=True)

	plt.scatter(x=itp_temp, y=itp_sal, marker=".", label="itp", s=1, alpha=0.5)
	plt.scatter(x=udash_temp, y=udash_sal, marker=".", label="udash", s=1, alpha=0.5)
	plt.scatter(x=argo_temp, y=argo_sal, marker=".", label="udash", s=1, alpha=0.5)
	plt.legend()
	plt.savefig(os.path.join(get_plot_dir(), "t_s.png"), dpi=500)
	plt.show()

	t_combined_df = ts_combined_df.drop_duplicates(["temp", "source"]).reset_index(drop=True)
	s_combined_df = ts_combined_df.drop_duplicates(["sal", "source"]).reset_index(drop=True)

	sns.histplot(data=t_combined_df, x="temp", hue="source")
	plt.savefig(os.path.join(get_plot_dir(), "all_temp.png"), dpi=1000)
	plt.show()

	sns.histplot(data=s_combined_df, x="sal", hue="source")
	plt.savefig(os.path.join(get_plot_dir(), "all_sal.png"), dpi=1000)
	plt.show()


def spatial_density(data: xr.Dataset, season: bool = False, decade: bool = False) -> None:
	if decade:
		decades = [
			("pre-2005", data.where(data.time.dt.year.load() < 2005, drop=True)),
			("post-2005", data.where(data.time.dt.year.load() >= 2005, drop=True)),
		]
	else:
		decades = [("all", data)]

	fig, axs = plt.subplots(
		nrows=len(decades),
		ncols=4,
		figsize=(10, 7),
		dpi=300,
		layout="constrained",
		subplot_kw={"projection": ccrs.NorthPolarStereo()},
	)

	# gs = plt.GridSpec(nrows=len(decades), ncols=4)
	ext = []

	for d, (dec_name, dec) in enumerate(decades):
		for i, data_seas in enumerate(dec.groupby(dec.time.dt.season)):
			ax = get_arctic_map(ax=axs[d, i])

			# Create the 2D histogram using hexbin
			hb = ax.hexbin(
				x=data_seas[1]["lon"],
				y=data_seas[1]["lat"],
				# C=data["temp"].values,
				gridsize=80,  # Adjust the gridsize to your preference
				cmap=cm.cm.dense,  # Choose the colormap you prefer
				transform=ccrs.PlateCarree(),
				bins="log",
				vmin=1,
				vmax=5e2,
			)

			# Add colorbar

			# Make colorbar height same as plot
			# ax_size = ax.get_position()
			# cbar.ax.set_position([ax_size.x1 + 0.1, ax_size.y0, 0.03, ax_size.height])

			ax.set_title(f"{data_seas[0]}")
		ext.append([axs[d, 0].get_window_extent().y0, axs[d, 0].get_window_extent().height])

	inv = fig.transFigure.inverted()
	upper_left = ext[0][0] + (ext[0][1] / fig.bbox.y1)
	upper_center = inv.transform((0.5, upper_left))
	lower_right = ext[1][0] + (ext[1][1] / fig.bbox.y1)
	lower_center = inv.transform((0.5, lower_right))

	plt.figtext(0.5, upper_center[1] + 0.0, "Pre 2005", va="center", ha="center", size=15)
	plt.figtext(0.5, lower_center[1] - 0.05, "Post 2005", va="center", ha="center", size=15)

	cbar = plt.colorbar(
		hb,
		ax=axs.ravel().tolist(),
		orientation="vertical",
		shrink=0.7,
		pad=0.05,
		label="Number of data points",
	)

	# plt.tight_layout()
	fig.suptitle("Histogram of the data density per season pre and post 2005", size=18)
	plt.savefig(os.path.join(get_plot_dir(), f"spatial_density_season_2005_d{len(decades)}.png"), dpi=1000)
	plt.show()


def main():
	from nereus.utils.directories import get_data_dir

	# ds = nereus.datasets.load_data()
	#
	# spatial_density(ds, season=True, decade=True)

	itps = xr.open_dataset(os.path.join(get_data_dir(), "itp", "cache", "itps_xr.nc"))
	udash = xr.open_dataset(os.path.join(get_data_dir(), "udash", "udash_xr.nc"))
	argos = xr.open_dataset(os.path.join(get_data_dir(), "argo", "argos_xr.nc"))

	all_spatial_xr(itps, udash, argos)


if __name__ == "__main__":
	main()
