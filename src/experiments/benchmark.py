"""Reusable CPU timing and serialized-model measurement helpers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np


def summarize_ns(values: list[int]) -> tuple[float, float]:
    if not values or any(value <= 0 for value in values):
        raise ValueError("timings must contain positive nanosecond values")
    values_ms = np.asarray(values, dtype=float) / 1_000_000.0
    return float(np.median(values_ms)), float(np.percentile(values_ms, 95))


def benchmark_callable(callable_: Callable[[], object], *, warmup: int = 100, repetitions: int = 300) -> tuple[float, float]:
    if warmup < 0 or repetitions < 1:
        raise ValueError("warmup must be non-negative and repetitions must be positive")
    for _ in range(warmup):
        callable_()
    timings = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        callable_()
        timings.append(time.perf_counter_ns() - start)
    return summarize_ns(timings)


def serialized_size_kb(path: str | Path) -> float:
    size = Path(path).stat().st_size
    if size <= 0:
        raise ValueError("serialized model must have positive size")
    return float(size / 1024.0)