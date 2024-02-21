from __future__ import annotations

import glob
import os
import shutil

import numpy as np
import pandas as pd
import xarray as xr

from tqdm import tqdm

from nereus import logger
from nereus.utils.directories import get_argo_dir

URL = "https://data-argo.ifremer.fr"


def extract_argo() -> None:
	argo_dir = get_argo_dir()
	logger.info("Extracting Argo")

	filename = glob.glob("DataSelection*.tar.gz", root_dir=argo_dir)
	print(filename)

	shutil.unpack_archive(
		filename=os.path.join(argo_dir, filename[0]),
		extract_dir=argo_dir,
		format="gztar"
	)


def load_all_argo() -> list:
	all_argo = [
		f for f in os.listdir(get_argo_dir())
		if f.endswith(".nc")
	]

	argos = []
	for f in tqdm(all_argo):
		argos.append(xr.open_dataset(os.path.join(get_argo_dir(), f)))
	return argos


def interp_argo(argo: pd.DataFrame, dims: list[str], x_inter, base_dim: str, **kwargs) -> pd.DataFrame:
	x_inter = np.arange(10, 760, 10)
	interp_itp = {
		"file": argo["file"].values[:len(x_inter)],  # So everything has the same length
		base_dim: x_inter
	}

	for dim in dims:
		if dim in argo:
			interp_itp[dim] = np.interp(x_inter, argo[base_dim].values, argo[dim].values)
		else:
			interp_itp[dim] = np.full(x_inter.shape, np.nan)
	return pd.DataFrame(interp_itp)


def preload() -> str:
	# check download
	# parse
	argos = parser_all_argo()
	logger.info("Parsed")
	processed_argo = []

	for itp in tqdm(argos):
		new_itp = interp_argo(itp, **kwargs)
		processed_argo.append(new_itp)

	logger.info("Concat")
	df_itps = pd.concat(processed_itps, ignore_index=True, keys=metadatas.index.get_level_values("file").to_list())

	logger.info("Join")
	df_itps = df_itps.join(metadatas, on="file")

	# df_itps.rename(columns=rename_col, inplace=True)

	logger.info("Caching")
	df_itps.to_parquet(os.path.join(get_argo_dir(), "argo_preprocessed.parquet"))

	# TODO: To xarray
	# itps_to_xr(df_itps)
	# save
	return df_itps


def main():
	extract_argo()


if __name__ == "__main__":
	main()
