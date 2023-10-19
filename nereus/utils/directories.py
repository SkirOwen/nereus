import os

from nereus.config import get_nereus_dir
from nereus.utils.file_ops import guarantee_existence


def get_data_dir() -> str:
	"""./data"""
	return guarantee_existence(os.path.join(get_nereus_dir(), "data"))


def get_itp_dir() -> str:
	"""./data/itp"""
	return guarantee_existence(os.path.join(get_data_dir(), "itp"))


def get_itp_extracted_dir() -> str:
	"""./data/itp/extracted"""
	return guarantee_existence(os.path.join(get_itp_dir(), "extracted"))
