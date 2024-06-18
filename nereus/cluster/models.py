from __future__ import annotations

import datetime
import os

from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import nereus
import nereus.datasets

from nereus.utils.decorator import panel_logger_decorator

from nereus import logger
from nereus.plots.plots import map_arctic_value, save_and_show
from nereus.utils.directories import get_data_dir, get_plot_dir
from nereus.processing.data_utils import get_scaler, pca_project


def plot_pca_score(data, comps, exp_vars, n_pc, titles):
	nbr_values = comps.shape[0]
	fig, axes = plt.subplots(nrows=1, ncols=nbr_values, figsize=(8, 6), dpi=300)
	for v in range(nbr_values):
		for i in range(n_pc):
			axes[v].plot(comps[v][i], data["pres"], label=f"Comp {i}, Exp_var:{exp_vars[v][i]:.2f}")

		axes[v].axvline(x=0, color="grey", linestyle="--")
		axes[v].set_title(titles[v])
		axes[v].set_xlabel("PCA Values")
		axes[v].invert_yaxis()
		axes[v].legend()
	axes[0].set_ylabel("Pressure (dbar)")
	plt.tight_layout()
	plt.show()


def gmm_metrics(fitted_model, data):
	# logger.info("aic")
	aic = fitted_model.aic(data)

	# logger.info("bic")
	bic = fitted_model.bic(data)

	# logger.info("sil")
	labels = fitted_model.predict(data)

	# logger.info(labels.size)
	sample_size = int(labels.size * 0.20) if labels.size > 10_000 else None
	sil = silhouette_score(data, labels, n_jobs=1, sample_size=sample_size)
	return aic, bic, sil


def fit_gmm(args):
	i, temp_sal_score = args
	gmm_model = GaussianMixture(n_components=i)
	gmm_model.fit(temp_sal_score)
	aic, bic, sil = gmm_metrics(gmm_model, data=temp_sal_score)
	return aic, bic, sil, gmm_model


def gmm(temp_sal_score_full, temp_sal_score_train, n_components: int = 4, random_state: None | int = None):
	model = GaussianMixture(n_components=n_components, random_state=random_state)
	print(model.n_components)

	model.fit(temp_sal_score_train)
	transformed_data = model.predict(temp_sal_score_full)
	transformed_proba = model.predict_proba(temp_sal_score_full)

	return transformed_data, model, transformed_proba


def gmm_benchmark(temp_sal_score: np.ndarray, max_components: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	tasks = [(i, temp_sal_score) for i in range(2, max_components)]
	results = []

	with Pool(6) as pool, tqdm(total=max_components, desc="Gaussians", position=1, leave=False) as pbar:
		for result in pool.imap(fit_gmm, tasks):
			results.append(result)
			pbar.update(1)

	aic, bic, silhouette_scores, models = zip(*results)
	return aic, bic, silhouette_scores


def plot_AIC_BIC_Si(aic: np.ndarray, bic: np.ndarray, silhouette_scores: np.ndarray, components: int = 20) -> None:
	metrics = {
		"AIC": aic,
		"BIC": bic,
		"Silhouette coefficient": silhouette_scores
	}

	gradients = {
		"AIC gradient": np.diff(aic),
		"BIC gradient": np.diff(bic),
		"Silhouette coefficient gradient": np.diff(silhouette_scores)
	}

	fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 11), dpi=300)

	for i, metric in enumerate(metrics):
		metric_data = metrics[metric]
		gradient_data = gradients[f"{metric} gradient"]

		# Plot metrics
		ax_metric = axes[i, 0]
		sns.lineplot(x=range(2, components), y=metric_data, ax=ax_metric, marker="o")
		ax_metric.set_xticks(range(2, components))
		ax_metric.set_ylabel(metric)
		ax_metric.grid()

		# Plot gradient
		ax_gradient = axes[i, 1]
		sns.lineplot(x=range(2, components - 1), y=gradient_data, ax=ax_gradient, marker="o")
		ax_gradient.set_xticks(range(2, components - 1))
		ax_gradient.set_ylabel(f"{metric} gradient")
		ax_gradient.grid()

		# Set x labels for the last row
		if i == 2:
			ax_metric.set_xlabel("Number of components")
			ax_gradient.set_xlabel("Number of components")

	plt.subplots_adjust(wspace=0.4, hspace=0.3)

	save_and_show(
		filename=f"AIC-BIC-Sil_gmm-{datetime.datetime.now().strftime('%Y-%m-%d@%H-%M-%S')}",
		directory=get_plot_dir(),
	)


def get_temp_sal_score(ds: xr.Dataset, n_pc: int, scaler_temp, scaler_sal) -> tuple:
	temp = ds["temp"].values
	sal = ds["sal"].values

	logger.info("PCA score")

	pca_projection_temp, comp_temp, exp_var_temp = pca_project(temp, n_pc=n_pc, scaler=scaler_temp)
	pca_projection_sal, comp_sal, exp_var_sal = pca_project(sal, n_pc=n_pc, scaler=scaler_sal)

	temp_sal_comps = np.array([comp_temp, comp_sal])
	temp_sal_exp_vars = np.array([exp_var_temp, exp_var_sal])
	temp_sal_score = np.concatenate((pca_projection_temp, pca_projection_sal), axis=1)

	return temp_sal_comps, temp_sal_exp_vars, temp_sal_score


def plot_mean_profile_allinone(ds_fit, cmap: list, variables: list[str] | None = None) -> None:
	"""
	Plot mean profile for each class in a single figure.
	"""

	unique_labels = np.unique(ds_fit.label.values)
	num_classes = len(unique_labels)
	variables = ["temp", "sal"] if variables is None else variables

	fig, axs = plt.subplots(1, len(variables), figsize=(8, 6), dpi=300)
	# Define your own color list

	for i in range(num_classes):
		ds_class = ds_fit.where(ds_fit.label == i, drop=True)
		if len(ds_class.profile) == 0:
			logger.warning("Skipping a class, make sure your training data is representative of your the entire dataset")
			continue

		for ax, var in zip(axs, variables):
			# Compute quantiles
			qua5 = np.quantile(ds_class[var], 0.05, axis=0)
			qua50 = np.quantile(ds_class[var], 0.50, axis=0)
			qua95 = np.quantile(ds_class[var], 0.95, axis=0)

			ax.plot(qua50, ds_fit["pres"].values, c=cmap[i], label=f"Class {i}")
			ax.fill_betweenx(ds_fit["pres"].values, qua5, qua95, color=cmap[i], alpha=0.5)
			ax.legend(loc="lower right")
			ax.set_ylim([np.min(ds_fit["pres"].values), np.max(ds_fit["pres"].values)])
			ax.set_xlabel("Temperature (°C)" if var == "temp" else "Salinity (g/kg)")
			ax.invert_yaxis()

	axs[0].set_ylabel("Pressure (dbar)")

	plt.tight_layout()

	plt.savefig(
		os.path.join(
			get_plot_dir(), f"mean_{num_classes}_profiles-{datetime.datetime.now().strftime('%Y-%m-%d@%H-%M-%S')}.png"
		)
	)
	plt.show()


def run(benchmark, n_pc, n_gmm, ensemble=False):
	logger.info("Loading full data")
	ds_full = nereus.datasets.load_data().load()
	ds_full = ds_full.dropna(dim="profile", subset=["temp", "sal"], how="any")
	# ds_full = _clean_data(ds_full)

	logger.info("Loading train data")
	ds = xr.open_dataset(os.path.join(get_data_dir(), "train_ds_100000_100000.nc")).load()
	ds = ds.dropna(dim="profile", subset=["temp", "sal"], how="any")
	# ds = _clean_data(ds)

	logger.info("Scale data")
	scaler_temp = get_scaler(ds_full["temp"].values)
	scaler_sal = get_scaler(ds_full["sal"].values)

	logger.info("PCA score")
	comps, exp_vars, temp_sal_score = get_temp_sal_score(ds, n_pc, scaler_temp, scaler_sal)
	comps_full, exp_vars_full, temp_sal_score_full = get_temp_sal_score(ds_full, n_pc, scaler_temp, scaler_sal)

	logger.info("Plot PCA")
	plot_pca_score(ds, comps, exp_vars, n_pc, titles=["temp_train", "sal_train"])

	logger.info("GMM")
	transformed_data, model, transformed_proba = gmm(temp_sal_score_full, temp_sal_score, n_components=n_gmm)
	logger.info("Done, merging")
	ds_full["label"] = ("profile", transformed_data)
	metrics_full = gmm_metrics(model, data=temp_sal_score_full)
	print(metrics_full)

	if benchmark:
		max_comp = 20
		logger.info(f"Calculate metrics for {max_comp}")
		aic, bic, sil = gmm_benchmark(temp_sal_score, max_components=max_comp)
		plot_AIC_BIC_Si(aic, bic, sil)
	else:
		pass

	if ensemble:
		aics, bics, sils = ensemble_model(train_data=temp_sal_score, full_data=temp_sal_score_full)
		plot_ensemble(aics, bics, sils)

	return ds_full


def ensemble_model(train_data, full_data, max_comp_per_run: int = 20, nbr_run: int = 2):
	metrics = []
	for i in tqdm(range(nbr_run), desc="Ensemble", position=0):
		metric = gmm_benchmark(train_data, max_comp_per_run)
		metrics.append(metric)
	aic, bic, sil = zip(*metrics)
	return np.array(aic), np.array(bic), np.array(sil)


def plot_ensemble(aic_ensemble, bic_ensemble, silhouette_scores_ensemble):
	# Calculate means and standard deviations
	metrics = {
		'AIC': (np.mean(aic_ensemble, axis=0), np.std(aic_ensemble, axis=0)),
		'BIC': (np.mean(bic_ensemble, axis=0), np.std(bic_ensemble, axis=0)),
		'Silhouette coefficient': (
		np.mean(silhouette_scores_ensemble, axis=0), np.std(silhouette_scores_ensemble, axis=0))
	}

	fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 11), dpi=300)
	x_range = range(2, 20)
	x_range_grad = range(2, 19)  # 20 - 1

	for i, (label, (mean, std)) in enumerate(metrics.items()):
		# Plot metric with error bars

		sns.lineplot(ax=axes[i, 0], x=x_range, y=mean, marker="o", label=label, errorbar='sd')
		axes[i, 0].fill_between(x_range, mean - std, mean + std, alpha=0.3)
		axes[i, 0].set_xticks(x_range)
		axes[i, 0].set_ylabel(label)
		axes[i, 0].grid()

		# Calculate and plot metric gradient with error bars
		grad_mean = np.diff(mean)
		grad_std = np.diff(std)

		sns.lineplot(ax=axes[i, 1], x=x_range_grad, y=grad_mean, marker="o", label=label, errorbar='sd')
		axes[i, 1].fill_between(x_range_grad, grad_mean - grad_std, grad_mean + grad_std, alpha=0.3)
		axes[i, 1].set_xticks(x_range_grad)
		axes[i, 1].set_ylabel(f"{label} gradient")
		axes[i, 1].grid()

		if i == 2:  # Set x labels for the last row
			axes[i, 0].set_xlabel('Number of components')
			axes[i, 1].set_xlabel('Number of components')

	plt.subplots_adjust(wspace=0.4, hspace=0.3)
	plt.show()

def _clean_data(ds, ):
	ds = ds.dropna(dim="profile", subset=["temp", "sal"], how="any")
	ds = ds.where(~(ds.temp > 25), drop=True)
	ds = ds.where((ds.sal > 15), drop=True)
	# ds = ds.where(np.logical_and(ds.sal > 25, ds.sal < 27), drop=True).sel(pres=slice(350, None), drop=True)
	drop_ds = ds.sel(pres=slice(350, None)).where(
		np.logical_and(
			ds.sel(pres=slice(350, None)).sal > 25,
			ds.sel(pres=slice(350, None)).sal < 27,
		), drop=True
	)
	ds = ds.drop_sel(profile=drop_ds.profile)
	return ds


def main():
	colour_palette = [
		"#cc6677",  # red              pinkish / dusty rose
		"#ddcc77",  # brown-orange     sand
		"#117733",  # dark green        darkish green
		"#88ccee",  # light blue
		"#44aa99",  # tesl blue         blue-green
		"#882255",  # dark magenta     red purple
		"#332288",  # deep blue        dark royal blue
		"#aa4499",  # purple
	]

	n_pc = 3
	n_gmm = 6
	ds_full = run(benchmark=True, n_pc=n_pc, n_gmm=n_gmm, ensemble=False)

	# plot_mean_profile_allinone(ds_full, cmap=colour_palette)
	# map_arctic_value(
	# 	ds_full.to_dataframe(),
	# 	name=f"output_{n_pc}_comp_{n_gmm}_gmm-{datetime.datetime.now().strftime('%Y-%m-%d@%H-%M-%S')}",
	# 	hue="label",
	# 	s=5,
	# 	palette=colour_palette
	# )


if __name__ == "__main__":
	main()
