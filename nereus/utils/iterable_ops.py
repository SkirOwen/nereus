from __future__ import annotations

import itertools as it

from typing import Iterable, Iterator


def skipwise(iterable: Iterable, step: int, start: None | int = None, stop: None | int = None) -> Iterator:
	pairwise_iter = it.pairwise(iterable)
	return it.islice(pairwise_iter, start, stop, step)
