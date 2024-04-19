from __future__ import annotations

import datetime
import os

from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import nereus

from nereus import logger
from nereus.utils.directories import *


def get_scaler(value):
	scaler = StandardScaler()
	scaler = scaler.fit(value)
	return scaler


def pca_score(value, n_pc: int, scaler):
	scaled_value = scaler.transform(value)

	pca_model = PCA(n_components=n_pc)
	fitted_value = pca_model.fit(scaled_value)

	comp = fitted_value.components_
	exp_var = fitted_value.explained_variance_ratio_

	# score = scaled_value @ comp.T
	score = pca_model.transform(scaled_value)

	return score, comp, exp_var


def plot_pca_score(data, comps, exp_vars, n_pc, titles):
	nbr_values = comps.shape[0]
	fig, axes = plt.subplots(nrows=1, ncols=nbr_values, figsize=(8, 6), dpi=300)
	for v in range(nbr_values):
		for i in range(n_pc):
			axes[v].plot(comps[v][i], data["pres"], label=f"Comp {i}, Exp_var:{exp_vars[v][i]:.2f}")
		axes[v].axvline(x=0, color="grey", linestyle="--")
		axes[v].set_title(titles[v])
		axes[v].set_xlabel("PCA Values")
		if v == 0:
			axes[v].set_ylabel("Pressure (dbar)")
		axes[v].invert_yaxis()
		# set the yticklabels to its absolute values
		axes[v].legend()
	plt.tight_layout()
	plt.show()


def fit_gmm(args):
	i, temp_sal_score = args
	gmm_model = GaussianMixture(n_components=i, covariance_type="full")
	gmm_model.fit(temp_sal_score)
	aic = gmm_model.aic(temp_sal_score)
	bic = gmm_model.bic(temp_sal_score)
	labels = gmm_model.predict(temp_sal_score)
	sil = silhouette_score(temp_sal_score, labels)
	return aic, bic, sil, gmm_model


def cal_AIC_BIC_Si(temp_sal_score: np.ndarray, max_components: int = 20):
	with Pool(6) as pool:
		results = list(
			tqdm(
				pool.imap(
					fit_gmm, [(i, temp_sal_score) for i in range(2, max_components)]
				),
				total=max_components - 2,
				desc="Gaussian",
			)
		)

	aic, bic, silhouette_scores, models = zip(*results)
	return aic, bic, silhouette_scores


def plot_AIC_BIC_Si(aic, bic, silhouette_scores, max_components: int = 20):
	bic_grad = np.diff(bic)
	aic_grad = np.diff(aic)
	silhouette_scores_grad = np.diff(silhouette_scores)

	fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 11), dpi=300)

	axes[0, 0].plot(range(2, max_components), aic)
	axes[0, 0].set_xticks(range(2, max_components))
	# axes[0, 0].set_xlabel('Number of components')
	axes[0, 0].set_ylabel("AIC")
	axes[0, 0].grid()

	axes[0, 1].plot(range(2, max_components - 1), aic_grad)
	axes[0, 1].set_xticks(range(2, max_components - 1))
	# axes[0, 1].set_xlabel('Number of components')
	axes[0, 1].set_ylabel("AIC gradient")
	axes[0, 1].grid()

	axes[1, 0].plot(range(2, max_components), bic)
	axes[1, 0].set_xticks(range(2, max_components))
	# axes[1, 0].set_xlabel('Number of components')
	axes[1, 0].set_ylabel("BIC")
	axes[1, 0].grid()

	axes[1, 1].plot(range(2, max_components - 1), bic_grad)
	axes[1, 1].set_xticks(range(2, max_components - 1))
	# axes[1, 1].set_xlabel('Number of components')
	axes[1, 1].set_ylabel("BIC gradient")
	axes[1, 1].grid()

	axes[2, 0].plot(range(2, max_components), silhouette_scores)
	axes[2, 0].set_xticks(range(2, max_components))
	axes[2, 0].set_xlabel("Number of components")
	axes[2, 0].set_ylabel("Silhouette coefficient")
	axes[2, 0].grid()

	axes[2, 1].plot(range(2, max_components - 1), silhouette_scores_grad)
	axes[2, 1].set_xticks(range(2, max_components - 1))
	axes[2, 1].set_xlabel("Number of components")
	axes[2, 1].set_ylabel("Silhouette coefficient gradient")
	axes[2, 1].grid()

	plt.subplots_adjust(wspace=0.4, hspace=0.3)
	plt.show()


def gmm(temp_sal_score_full, temp_sal_score_train, n_components: int = 4):
	model = GaussianMixture(n_components=n_components, random_state=0)
	print(model.n_components)

	model.fit(temp_sal_score_train)
	transformed_data = model.predict(temp_sal_score_full)

	return transformed_data, model


def get_temp_sal_score(ds, n_pc, scaler_temp, scaler_sal):
	temp = ds["temp"].values
	sal = ds["sal"].values

	logger.info("PCA score")

	score_temp, comp_temp, exp_var_temp = pca_score(temp, n_pc=n_pc, scaler=scaler_temp)
	score_sal, comp_sal, exp_var_sal = pca_score(sal, n_pc=n_pc, scaler=scaler_sal)

	comps = np.array([comp_temp, comp_sal])
	exp_vars = np.array([exp_var_temp, exp_var_sal])
	temp_sal_score = np.concatenate((score_temp, score_sal), axis=1)

	return comps, exp_vars, temp_sal_score


def plot_mean_profile_allinone(ds_fit, cmap) -> None:
	"""
	Plot mean profile for each class in a single figure.

	Args:
		ds_fit: The input dataset.
	"""

	unique_labels = np.unique(ds_fit.label.values)
	num_classes = len(unique_labels)

	fig, axs = plt.subplots(1, 2, figsize=(8, 6), dpi=300)
	# Define your own color list

	for i in range(num_classes):
		ds_class = ds_fit.where(ds_fit.label == i, drop=True)

		qua5_temp = np.quantile(ds_class["temp"], 0.05, axis=0)
		qua95_temp = np.quantile(ds_class["temp"], 0.95, axis=0)
		qua50_temp = np.quantile(ds_class["temp"], 0.50, axis=0)

		qua5_salinity = np.quantile(ds_class["sal"], 0.05, axis=0)
		qua95_salinity = np.quantile(ds_class["sal"], 0.95, axis=0)
		qua50_salinity = np.quantile(ds_class["sal"], 0.50, axis=0)

		ax = axs[0]
		ax.plot(qua50_temp, ds_fit["pres"].values, c=cmap[i], label=f"Class {i}")
		ax.fill_betweenx(ds_fit["pres"].values, qua5_temp, qua95_temp, color=cmap[i], alpha=0.5)
		ax.legend(loc="lower right")
		ax.set_ylim([np.min(ds_fit["pres"].values), np.max(ds_fit["pres"].values)])
		ax.set_ylabel("Pressure (dbar)")
		# ax.set_yticklabels(np.abs(ax.get_yticks()).astype(int))
		ax.set_xlabel("Temperature (°C)")
		ax.invert_yaxis()

		ax = axs[1]
		ax.plot(qua50_salinity, ds_fit["pres"].values, c=cmap[i], label=f"Class {i}")
		ax.fill_betweenx(ds_fit["pres"].values, qua5_salinity, qua95_salinity, color=cmap[i], alpha=0.5)
		ax.set_ylim([np.min(ds_fit["pres"].values), np.max(ds_fit["pres"].values)])
		# ax.legend(loc='lower left')
		# ax.set_ylabel('Pressure')
		# ax.set_yticklabels(np.abs(ax.get_yticks()).astype(int))
		ax.set_ylabel("")
		ax.set_xlabel("Salinity (g/kg)")
		ax.invert_yaxis()

	plt.tight_layout()

	plt.savefig(
		os.path.join(
			get_plot_dir(), f"mean_{num_classes}_profiles-{datetime.datetime.now().strftime('%Y-%m-%d@%H-%M-%S')}.png"
		)
	)
	plt.show()


def run(benchmark, n_pc, n_gmm):
	logger.info("Loading full data")
	data_full = nereus.load_data().load()
	ds = data_full.dropna(dim="profile", subset=["temp", "sal"], how="any")
	ds = ds.where(~(ds.temp > 25), drop=True)
	ds = ds.where(~(ds.sal < 15), drop=True)
	ds.where(np.logical_and(ds.sal > 25, ds.sal < 27), drop=True).sel(pres=slice(350, None), drop=True)
	ds = ds.where(~(ds.sal < 15), drop=True)
	drop_ds = ds.sel(pres=slice(350, None)).where(
		np.logical_and(
			ds.sel(pres=slice(350, None)).sal > 25,
			ds.sel(pres=slice(350, None)).sal < 27,
		), drop=True
	)
	ds = ds.drop_sel(profile=drop_ds.profile, errors="ignore")
	ds_full = ds.dropna(dim="profile", subset=["temp", "sal"], how="any")

	logger.info("Loading train data")
	data = xr.open_dataset(os.path.join(get_data_dir(), "train_ds_10000_10000.nc")).load()
	ds = data.dropna(dim="profile", subset=["temp", "sal"], how="any")
	ds = ds.where(~(ds.temp > 25), drop=True)
	ds = ds.where(~(ds.sal < 15), drop=True)
	ds.where(np.logical_and(ds.sal > 25, ds.sal < 27), drop=True).sel(pres=slice(350, None), drop=True)
	ds = ds.where(~(ds.sal < 15), drop=True)
	drop_ds = ds.sel(pres=slice(350, None)).where(
		np.logical_and(
			ds.sel(pres=slice(350, None)).sal > 25,
			ds.sel(pres=slice(350, None)).sal < 27,
		), drop=True
	)
	ds = ds.drop_sel(profile=drop_ds.profile)
	ds = ds.dropna(dim="profile", subset=["temp", "sal"], how="any")

	scaler_temp = get_scaler(ds_full["temp"].values)
	scaler_sal = get_scaler(ds_full["sal"].values)

	comps, exp_vars, temp_sal_score = get_temp_sal_score(ds, n_pc, scaler_temp, scaler_sal)
	comps_full, exp_vars_full, temp_sal_score_full = get_temp_sal_score(ds_full, n_pc, scaler_temp, scaler_sal)

	logger.info("Plot PCA")
	plot_pca_score(ds, comps, exp_vars, n_pc, titles=["temp_train", "sal_train"])

	logger.info("GMM")
	transformed_data, model = gmm(temp_sal_score_full, temp_sal_score, n_components=n_gmm)
	ds_full["label"] = ("profile", transformed_data)

	if benchmark:
		max_comp = 20
		logger.info(f"Calculate metrics for {max_comp - 1}")
		aic, bic, sil = cal_AIC_BIC_Si(temp_sal_score, max_components=max_comp)
		plot_AIC_BIC_Si(aic, bic, sil, max_components=max_comp)

	return ds_full


def main():
	colour_palette = [
		"#cc6677",  # red              pinkish / dusty rose
		"#ddcc77",  # brown-orange     sand
		"#117733",  # dark green        darkish green
		"#88ccee",  # light blue
		"#44aa99",  # tesl blue         blue-green
		"#882255",  # dark magenta     red purple
		"#332288",  # deep blue        dark royal blue
	]

	n_pc = 2
	n_gmm = 4
	ds_full = run(benchmark=False, n_pc=n_pc, n_gmm=n_gmm)
	plot_mean_profile_allinone(ds_full, cmap=colour_palette)
	# map_arctic_value(
	# 	ds_full.to_dataframe(),
	# 	name=f"output_{n_pc}_comp_{n_gmm}_gmm-{datetime.datetime.now().strftime('%Y-%m-%d@%H-%M-%S')}",
	# 	hue="label",
	# 	s=5,
	# 	palette=colour_palette
	# )


if __name__ == "__main__":
	main()
