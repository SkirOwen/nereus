from __future__ import annotations

import glob
import os
import shutil

import xarray as xr

from tqdm import tqdm

from nereus import logger
from nereus.utils.downloader import downloader
from nereus.utils.directories import get_argo_dir

URL = "https://data-argo.ifremer.fr"


def download_argo():
	pass


def extract_argo():
	argo_dir = get_argo_dir()
	logger.info("Extracting Argo")

	filename = glob.glob("DataSelection*.tar.gz", root_dir=argo_dir)
	print(filename)

	shutil.unpack_archive(
		filename=os.path.join(argo_dir, filename[0]),
		extract_dir=argo_dir,
		format="gztar"
	)


def load_all_argo():
	all_argo = [
		f for f in os.listdir(get_argo_dir())
		if f.endswith(".nc")
	]

	dss = []
	for f in tqdm(all_argo):
		dss.append(xr.open_dataset(os.path.join(get_argo_dir(), f)))
	return dss


def main():
	extract_argo()


if __name__ == "__main__":
	main()
