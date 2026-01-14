from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import os
from typing import TypeVar, cast

T = TypeVar("T")
R = TypeVar("R")


def resolve_jobs(jobs: int | None, *, env_var: str | None = "SPR_THREADS") -> int:
    if jobs is None:
        env_jobs = os.getenv(env_var) if env_var else None
        if env_jobs:
            try:
                jobs = int(env_jobs)
            except ValueError:
                jobs = 0
        else:
            jobs = 0
    if jobs < 0:
        raise ValueError("jobs must be >= 0")
    if jobs == 0:
        cpu_count = os.cpu_count() or 1
        return max(cpu_count - 1, 1)
    return jobs


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int,
) -> list[R]:
    if max_workers <= 1:
        return [func(item) for item in items]
    items_list = list(items)
    results: list[R | None] = [None] * len(items_list)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(func, item): idx
            for idx, item in enumerate(items_list)
        }
        for future in as_completed(future_map):
            idx = future_map[future]
            results[idx] = future.result()
    return cast(list[R], results)


def parallel_map_process(
    func: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int,
) -> list[R]:
    if max_workers <= 1:
        return [func(item) for item in items]
    items_list = list(items)
    results: list[R | None] = [None] * len(items_list)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(func, item): idx
            for idx, item in enumerate(items_list)
        }
        for future in as_completed(future_map):
            idx = future_map[future]
            results[idx] = future.result()
    return cast(list[R], results)


@contextmanager
def limit_blas_threads(threads: int) -> Iterator[None]:
    keys = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    prev = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ[key] = str(threads)
    try:
        yield
    finally:
        for key, value in prev.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
