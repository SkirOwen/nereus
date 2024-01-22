from __future__ import annotations

import glob
import os
import shutil

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
		filename=os.path.join(argo_dir, "filename"),
		extract_dir=argo_dir,
		format="gztar"
	)


def main():
	extract_argo()


if __name__ == "__main__":
	main()
