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

	fig = plt.figure(figsize=(20, 20))
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


def main():
	pass


if __name__ == "__main__":
	main()
