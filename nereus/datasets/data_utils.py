from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def convert_to_xr(df: pd.DataFrame, coords: list[str]) -> xr.Dataset:
	unique_coords = df.drop_duplicates("profile").set_index("profile")[coords]
	df.set_index(["profile", "pres"], inplace=True)

	ds = xr.Dataset.from_dataframe(df)
	co_ds = xr.Dataset.from_dataframe(unique_coords).set_coords(coords)
	ds = xr.merge([ds.drop_vars(coords), co_ds])
	return ds


def main():
	pass


if __name__ == "__main__":
	main()
