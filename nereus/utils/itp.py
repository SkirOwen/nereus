from __future__ import annotations

import os.path
import shutil
import ssl
import urllib.request

from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from nereus import logger

from nereus.utils.downloader import downloader
from nereus.utils.file_ops import calculate_md5
from nereus.utils.directories import get_itp_dir, get_itp_extracted_dir

URL = "https://scienceweb.whoi.edu/itp/data/"
MD5_URL = "https://scienceweb.whoi.edu/itp-md5sums/MD5SUMS"


def get_filenames_from_url(url: str) -> list[str]:
	"""
	Gets the filename list for files ending with 'final.tar.Z' from an archive-like URL.

	Parameters
	----------
	url : str
		The URL from which to scrape the filenames.

	Returns
	-------
	list[str]
		A list of filenames ending with 'final.tar.Z'. If no such filenames are found,
		an empty list is returned. If an error occurs, an empty list is returned and the error
		is logged.
	"""

	ssl._create_default_https_context = ssl._create_unverified_context
	req = urllib.request.Request(url)
	req.add_header(
		'user-agent',
		'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'
	)
	response = urllib.request.urlopen(req)
	html_content = response.read().decode("utf-8")

	lines = html_content.split("\n")

	file_names = []
	for line in lines:
		if "final.zip" in line:
			start_idx = line.find('href="') + 6  # Find the start index of the filename
			end_idx = line.find('"', start_idx)  # Find the end index of the filename
			if start_idx != -1 and end_idx != -1:  # Check that both indices were found
				file_name = line[start_idx:end_idx]
				file_names.append(file_name)

	return file_names


def get_md5sum_dict() -> dict[str, str]:
	"""
	Function to download and return the md5 sums of the ipts as a dictionary.

	Returns
	-------
	dict,
		Keys are the name of the file and the value is their md5 check sum.
	"""
	md5sum_filepath = os.path.join(get_itp_dir(), "MD5SUMS")
	if not os.path.exists(md5sum_filepath):
		downloader([MD5_URL], get_itp_dir())

	hash = calculate_md5(md5sum_filepath)
	if not(hash == "67ecdfe4bac8a5fd277bdf67cb59d7b6"):
		logger.info("The md5 of the md5 file did not match.")

	with open(md5sum_filepath, "r") as f:
		lines = f.readlines()

	md5_dict = {
		filename: value for line in lines for filename, value in [line.split()]
	}

	return md5_dict


def download_itp(main_url: str, files: None | list[str] = None, override: bool = False) -> None:
	"""
	Downloads files with the extension 'final.tar.Z' from the specified main URL.

	Parameters
	----------
	main_url : str
		The main URL where the files are hosted.
	files : None | list[str], optional
		A list of filenames to download. If `None`, the function will scrape the filenames
		using `get_filenames_from_url` function. Default is `None`.
	override : bool, optional
		Whether to override existing files with the same name. Default is `False`.
	"""
	if files is None:
		files = get_filenames_from_url(main_url)

	itp_dir = get_itp_dir()

	urls = [main_url + f for f in files]
	downloader(urls, itp_dir, override=override)


def extract_itp(file: str, target_directory: None | str = None) -> None:
	"""

	Parameters
	----------
	file
	target_directory : str, optional
		Default will be ./data/itp/extracted

	Returns
	-------
	"""
	target_directory = get_itp_extracted_dir() if target_directory is None else target_directory
	shutil.unpack_archive(filename=file, extract_dir=target_directory, format="zip")


def extract_all_itps(itp_dir: str, target_dir: None | str = None):
	all_itps = [
		f for f in os.listdir(itp_dir)
		if f.endswith(".zip")
	]

	with ThreadPoolExecutor(max_workers=4) as pool:
		for task, itp in enumerate(tqdm(all_itps)):
			itp_filepath = os.path.join(itp_dir, itp)
			pool.submit(extract_itp, itp_filepath, target_dir)
		logger.info("All ITPs have been extracted.")


def main():
	download_itp(main_url=URL)


if __name__ == "__main__":
	main()
