from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_scaler(value):
	scaler = StandardScaler()
	scaler = scaler.fit(value)
	return scaler


def pca_project(value, n_pc: int, scaler) -> tuple:
	scaled_value = scaler.transform(value)

	pca_model = PCA(n_components=n_pc)
	fitted_value = pca_model.fit(scaled_value)

	comp = fitted_value.components_
	exp_var = fitted_value.explained_variance_ratio_

	# score = scaled_value @ comp.T
	pca_projection = pca_model.transform(scaled_value)

	return pca_projection, comp, exp_var
