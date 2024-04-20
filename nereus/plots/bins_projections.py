from __future__ import annotations

import os
import warnings

import cartopy.crs as ccrs
import cmocean as cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, RadioButtons, Slider

from nereus import logger
from nereus.cluster.training_data import prepare_training_data
from nereus.datasets.merged import load_data
from nereus.plots.plots import get_arctic_map
from nereus.processing.coordinates_utils import grid_bin, grid_bin_coord
from nereus.utils.directories import get_data_dir


mpl.use("TkAgg")


def map_groups_proj(latlon_groups, proj):
	warnings.simplefilter("ignore", UserWarning)

	fig, ax1 = get_arctic_map()

	# Grid so I can resize the all the axes
	gs = GridSpec(2, 1, figure=fig)
	ax1.set_position(gs[0].get_position(fig))

	g_len = []

	# for group_key in latlon_groups.groups:
	for group in latlon_groups:
		try:
			# group = latlon_groups.get_group(group_key)
			# print(f"Group {group_key}: {len(group)} rows")

			g_len.append(len(group[1]))

			g = sns.scatterplot(
				data=group[1],
				x="x",
				y="y",
				s=1,
				ax=ax1,
				transform=proj,
				markers="h",
			)
		except KeyError:
			# Handle the KeyError gracefully
			group_key = ""
			print(f"No data points for group {group_key}")

	ax2 = fig.add_subplot(gs[1])
	line = ax2.scatter(list(range(len(g_len))), sorted(g_len))

	return fig, ax1, ax2, g, line


def get_g_(latlon_groups) -> list:
	g_len = [len(group[1]) for group in latlon_groups]
	return g_len


def plot_g(latlon_groups, ax1, proj) -> None:
	warnings.simplefilter("ignore", UserWarning)
	for group in latlon_groups:
		g = sns.scatterplot(
			data=group[1],
			x="x",
			y="y",
			s=0.1,
			ax=ax1,
			transform=proj,
			markers="h",
		)


def plot_hex(df, ax, **kwargs):
	hb = ax.hexbin(
		x=df["lon"],
		y=df["lat"],
		# C=data["temp"].values,
		gridsize=(80),  # Adjust the gridsize to your preference
		cmap=cm.cm.dense,  # Choose the colormap you prefer
		transform=ccrs.PlateCarree(),
		bins="log",
		**kwargs,
	)
	return hb


def get_proj(val):
	match val:
		case "north":
			proj = ccrs.NorthPolarStereo()
		case "lambert":
			proj = ccrs.LambertAzimuthalEqualArea(central_latitude=90.0)
		case "AlbersEqualArea":
			proj = ccrs.AlbersEqualArea(central_latitude=90.0)
		case "EqualEarth":
			proj = ccrs.EqualEarth()
		case "Nearside":
			proj = ccrs.NearsidePerspective(central_latitude=90.0)
		case "platecarree":
			proj = ccrs.PlateCarree()
	return proj


def interactive(data):
	fig = plt.figure(figsize=(8, 8), dpi=300)
	gs = GridSpec(2, 3, figure=fig)

	ax1 = get_arctic_map(ax=fig.add_subplot(gs[0], projection=ccrs.NorthPolarStereo()))
	ax2 = fig.add_subplot(gs[3:])
	ax3 = get_arctic_map(ax=fig.add_subplot(gs[1], projection=ccrs.NorthPolarStereo()))
	ax4 = get_arctic_map(ax=fig.add_subplot(gs[2], projection=ccrs.NorthPolarStereo()))

	x_step_start = 100_000
	y_step_start = 100_000

	# fig, ax1, ax2, g, line = map_groups_proj(latlon_groups, proj=ccrs.NorthPolarStereo())
	latlon_groups = grid_bin_coord(data, ccrs.NorthPolarStereo(), x_step_start, y_step_start)
	ds_train = prepare_training_data(
		x_step_start, y_step_start, quantile4T=0.95, ds_cleaned=data, proj=ccrs.NorthPolarStereo()
	)

	# ax1
	plot_g(latlon_groups, ax1, proj=ccrs.NorthPolarStereo())

	g_len = get_g_(latlon_groups)
	ax2.plot(list(range(len(g_len))), sorted(g_len), "ok")

	# ax3, ax4
	hb3 = plot_hex(data.to_dataframe(), ax3, vmin=10)

	vmax = hb3.get_array().max()
	vmin = 10
	hb4 = plot_hex(ds_train.to_dataframe(), ax4, vmin=vmin, vmax=vmax)

	cax = fig.add_axes([0.92, 0.55, 0.02, 0.4])
	max_count = max(hb3.get_array().max(), hb4.get_array().max())
	cbar = fig.colorbar(hb4, cax=cax)
	cbar.set_label("Colorbar Label")

	axx = fig.add_axes([0.02, 0.50, 0.0225, 0.45])
	axy = fig.add_axes([0.08, 0.50, 0.0225, 0.45])
	x_slider = Slider(
		ax=axx,
		label="x",
		valmin=0,
		valmax=500_000,
		valinit=10_000,
		valstep=10_000,
		orientation="vertical",
	)
	y_slider = Slider(
		ax=axy,
		label="y",
		valmin=0,
		valmax=500_000,
		valinit=10_000,
		valstep=10_000,
		orientation="vertical",
	)

	def update(val):
		proj = get_proj(radio.value_selected)
		scale = 10_000 if proj == ccrs.PlateCarree() else 1
		latlon_groups = grid_bin(data, x_step=x_slider.val, y_step=y_slider.val, scale=scale)
		ds_train = prepare_training_data(x_slider.val, y_slider.val, quantile4T=0.95, ds_cleaned=data, proj=proj)

		g_len = get_g_(latlon_groups)

		# clearing ax1 but keeping map
		for artist in ax1.lines + ax1.collections:
			artist.remove()
		plot_g(latlon_groups, ax1, proj)

		# clearing ax2
		ax2.clear()
		ax2.plot(list(range(len(g_len))), sorted(g_len), "ok")

		for artist in ax4.lines + ax4.collections:
			artist.remove()
		hb4 = plot_hex(ds_train.to_dataframe(), ax4, vmin=vmin, vmax=vmax)

		cbar = fig.colorbar(hb4, cax=cax)
		cbar.set_label("Colorbar Label")

		fig.canvas.draw_idle()

	#
	# x_slider.on_changed(update)
	# y_slider.on_changed(update)

	recalc_button_ax = fig.add_axes([0.1, 0.90, 0.1, 0.04])
	button = Button(recalc_button_ax, "Calc", hovercolor="0.975")

	button.on_clicked(update)

	button_ax = fig.add_axes([0.80, 0.50, 0.1, 0.45])
	button_ax.set_axis_off()
	radio = RadioButtons(button_ax, ["north", "lambert", "AlbersEqualArea", "EqualEarth", "Nearside", "platecarree"])

	def change_proj(val):
		proj = get_proj(val)
		scale = 10_000 if proj == ccrs.PlateCarree() else 1
		latlon_groups = grid_bin_coord(data, proj, x_step=x_slider.val, y_step=y_slider.val, scale=scale)
		ds_train = prepare_training_data(x_slider.val, y_slider.val, quantile4T=0.95, ds_cleaned=data, proj=proj)

		g_len = get_g_(latlon_groups)

		# clearing ax1 but keeping map
		for artist in ax1.lines + ax1.collections:
			artist.remove()
		plot_g(latlon_groups, ax1, proj)

		# clearing ax2
		ax2.clear()
		ax2.plot(list(range(len(g_len))), sorted(g_len), "ok")

		for artist in ax4.lines + ax4.collections:
			artist.remove()
		hb4 = plot_hex(ds_train.to_dataframe(), ax4, vmin=vmin, vmax=vmax)

		cbar = fig.colorbar(hb4, cax=cax)
		cbar.set_label("Colorbar Label")

		fig.canvas.draw_idle()

	radio.on_clicked(change_proj)

	save_button_pos = fig.add_axes([0.60, 0.90, 0.1, 0.04])
	save_button = Button(save_button_pos, "save", hovercolor="0.975")

	def save(val):
		ds_train.to_netcdf(
			os.path.join(get_data_dir(), f"train_ds_{x_slider.val}_{y_slider.val}.nc"),
			format="NETCDF4",
			engine="h5netcdf",
		)
		logger.info("Saved")

	save_button.on_clicked(save)

	plt.show()


def main():
	data = load_data().load()
	interactive(data)


if __name__ == "__main__":
	main()
