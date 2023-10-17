from __future__ import annotations

import json
import os.path
import urllib.request
import urllib.error

from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPResponse
from tqdm.auto import tqdm

from typing import Iterable, Generator, Sequence

from nereus import logger

CHUNK_SIZE = 1024


def _get_response_size(resp: HTTPResponse) -> None | int:
	"""
	Get the size of the file to download
	"""
	try:
		return int(resp.info()["Content-length"])
	except (ValueError, KeyError, TypeError):
		return None


def _get_chunks(resp: HTTPResponse) -> Generator[bytes, None]:
	"""
	Generator of the chunks to download
	"""
	while True:
		chunk = resp.read(CHUNK_SIZE)
		if not chunk:
			break
		yield chunk


def _get_response(url: str) -> HTTPResponse:
	try:
		response = urllib.request.urlopen(url)
	except urllib.error.HTTPError:
		import base64
		from http.cookiejar import CookieJar
		cj = CookieJar()
		opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
		request = urllib.request.Request(url)

		# user, password = _credential_helper(base_url=os.path.dirname(url))

		# base64string = base64.b64encode((user + ":" + password).encode("ascii"))
		# request.add_header("Authorization", "Basic {}".format(base64string.decode("ascii")))
		response = opener.open(request)
	except urllib.error.URLError:
		# work around to be able to dl the 10m coastline without issue
		import ssl
		ssl._create_default_https_context = ssl._create_unverified_context
		req = urllib.request.Request(url)
		req.add_header(
			'user-agent',
			'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'
		)
		response = urllib.request.urlopen(req)
	return response


def url_download(url: str, path: str, task: int = 1, total: int = 1) -> None:
	"""
	Download an url to a local file

	See Also
	--------
	downloader : Downloads multiple url in parallel.
	"""
	logger.info(f"Downloading: '{url}' to {path}")
	response = _get_response(url)
	chunks = _get_chunks(response)
	pbar = tqdm(
		desc=f"[{task}/{total}] Requesting {os.path.basename(url)}",
		unit="B",
		total=_get_response_size(response),
		unit_scale=True,
		# format to have current/total size with the full unit, e.g. 60kB/6MB
		# https://github.com/tqdm/tqdm/issues/952
		bar_format="{l_bar}{bar}| {n_fmt}{unit}/{total_fmt}{unit}"
		           " [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
	)
	with pbar as t:
		with open(path, "wb") as file:
			for chunk in chunks:
				file.write(chunk)
				t.update(len(chunk))
			# if done_event.is_set():
			# 	return
	logger.debug(f"Downloaded in {path}")


def downloader(urls: Sequence[str], root: str, override: bool = False):
	"""
	Downloader to download multiple files.
	"""
	if isinstance(urls, str):
		urls = [urls]

	with ThreadPoolExecutor(max_workers=4) as pool:
		root = os.path.abspath(root)
		for task, url in enumerate(urls, start=1):
			filename = url.split("/")[-1]
			filename = filename.split("?")[0]   # Removing HTML tag/option
			target_path = os.path.join(root, filename)

			if not os.path.exists(target_path) or override:
				# TODO: when file present it should only skip if checksum matches, if checksum_check is done
				pool.submit(url_download, url, target_path, task, total=len(urls))
			else:
				logger.info(f"Skipping {filename} as already present in {root}")


# future update:
# using rich
# inside a 'context' box:
# top pbar is for the url in urls
# inside, individual pbar for all the downloads
# see nala package on ubuntu


def main():
	url = [
		'https://imgs.xkcd.com/comics/overlapping_circles.png',
	]
	response = _get_response(url[0])
	print(response)

	# target_dist = get_download_dir()
	# downloader(url, target_dist)


if __name__ == "__main__":
	main()
