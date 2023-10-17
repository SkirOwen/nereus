from __future__ import annotations

import ssl
import urllib.request

from nereus.utils.downloader import downloader
from nereus.utils.directories import get_itp_dir

URL = "https://scienceweb.whoi.edu/itp/data/"


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
		if 'final.tar.Z' in line:
			start_idx = line.find('href="') + 6  # Find the start index of the filename
			end_idx = line.find('"', start_idx)  # Find the end index of the filename
			if start_idx != -1 and end_idx != -1:  # Check that both indices were found
				file_name = line[start_idx:end_idx]
				file_names.append(file_name)

	return file_names


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


def main():
	download_itp(main_url=URL)


if __name__ == "__main__":
	main()
