from __future__ import annotations

import warnings

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.gridspec import GridSpec
from matplotlib.widgets import RadioButtons, Slider, Button

from nereus.plots.plots import get_arctic_map
from nereus.processing.coordinates_utils import change_coords, grid_bin
from nereus.datasets.merged import load


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


def get_g_(latlon_groups):
	g_len = [len(group[1]) for group in latlon_groups]
	return g_len


def plot_g(latlon_groups, ax1, proj):
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
	fig, ax1 = get_arctic_map()
	gs = GridSpec(2, 1, figure=fig)
	ax1.set_position(gs[0].get_position(fig))
	ax2 = fig.add_subplot(gs[1])
	ds = change_coords(data, proj=ccrs.NorthPolarStereo())

	# fig, ax1, ax2, g, line = map_groups_proj(latlon_groups, proj=ccrs.NorthPolarStereo())
	latlon_groups = grid_bin(ds, 100_000, 100_000)

	# ax1
	plot_g(latlon_groups, ax1, proj=ccrs.NorthPolarStereo())

	g_len = get_g_(latlon_groups)
	ax2.plot(list(range(len(g_len))), sorted(g_len), "ok")

	axx = fig.add_axes([0.02, 0.50, 0.0225, 0.45])
	axy = fig.add_axes([0.08, 0.50, 0.0225, 0.45])
	x_slider = Slider(
		ax=axx,
		label="x",
		valmin=0,
		valmax=500_000,
		valinit=10_000,
		valstep=10_000,
		orientation="vertical"
	)
	y_slider = Slider(
		ax=axy,
		label="y",
		valmin=0,
		valmax=500_000,
		valinit=10_000,
		valstep=10_000,
		orientation="vertical"
	)

	def update(val):
		proj = get_proj(radio.value_selected)
		scale = 10_000 if proj == ccrs.PlateCarree() else 1
		latlon_groups = grid_bin(data, x_step=x_slider.val, y_step=y_slider.val, scale=scale)
		g_len = get_g_(latlon_groups)

		# clearing ax1 but keeping map
		for artist in ax1.lines + ax1.collections:
			artist.remove()
		plot_g(latlon_groups, ax1, proj)

		# clearing ax2
		ax2.clear()
		ax2.plot(list(range(len(g_len))), sorted(g_len), "ok")
		fig.canvas.draw_idle()
	#
	# x_slider.on_changed(update)
	# y_slider.on_changed(update)

	recalc_button_ax = fig.add_axes([0.1, 0.75, 0.1, 0.04])
	button = Button(recalc_button_ax, 'Calc', hovercolor='0.975')

	button.on_clicked(update)

	button_ax = fig.add_axes([0.80, 0.50, 0.1, 0.45])
	button_ax.set_axis_off()
	radio = RadioButtons(button_ax, [
		"north",
		"lambert",
		"AlbersEqualArea",
		"EqualEarth",
		"Nearside",
		"platecarree"
	])

	def change_proj(val):
		proj = get_proj(val)
		data_remaped = change_coords(data, proj)
		scale = 10_000 if proj == ccrs.PlateCarree() else 1
		latlon_groups = grid_bin(data_remaped, x_step=x_slider.val, y_step=y_slider.val, scale=scale)
		g_len = get_g_(latlon_groups)

		# clearing ax1 but keeping map
		for artist in ax1.lines + ax1.collections:
			artist.remove()
		plot_g(latlon_groups, ax1, proj)

		# clearing ax2
		ax2.clear()
		ax2.plot(list(range(len(g_len))), sorted(g_len), "ok")
		fig.canvas.draw_idle()
		# fig.canvas.draw_idle()

	radio.on_clicked(change_proj)


def main():
	pass


if __name__ == "__main__":
	main()
