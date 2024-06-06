import os

from nereus.config import get_nereus_dir
from nereus.utils.file_ops import guarantee_existence


def get_data_dir() -> str:
	"""./data"""
	return guarantee_existence(os.path.join(get_nereus_dir(), "data"))


def get_itp_dir() -> str:
	"""./data/itp"""
	return guarantee_existence(os.path.join(get_data_dir(), "itp"))


def get_itp_cache_dir() -> str:
	"""./data/itp/cache"""
	return guarantee_existence(os.path.join(get_itp_dir(), "cache"))


def get_itp_extracted_dir() -> str:
	"""./data/itp/extracted"""
	return guarantee_existence(os.path.join(get_itp_dir(), "extracted"))


def get_udash_dir() -> str:
	"""./data/udash"""
	return guarantee_existence(os.path.join(get_data_dir(), "udash"))


def get_udash_extracted_dir() -> str:
	"""./data/udash/UDASH"""
	return guarantee_existence(os.path.join(get_udash_dir(), "UDASH"))


def get_argo_dir() -> str:
	"""./data/argo"""
	return guarantee_existence(os.path.join(get_data_dir(), "argo"))


def get_plot_dir() -> str:
	"""./plots"""
	return guarantee_existence(os.path.join(get_nereus_dir(), "plots"))


def get_log_dir() -> str:
	"""./logs"""
	return guarantee_existence(os.path.join(get_nereus_dir(), "logs"))
