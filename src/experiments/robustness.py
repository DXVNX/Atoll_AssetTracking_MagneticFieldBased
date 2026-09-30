"""Deterministic, trajectory-safe sensor corruption helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schemas import SENSOR_COLUMNS, validate_canonical_samples
from src.data.windows import resample_segments


def estimate_sampling_rate_hz(samples: pd.DataFrame) -> float:
    """Estimate the nominal rate from positive timestamp deltas."""
    validate_canonical_samples(samples)
    deltas = samples.groupby("trajectory_id", sort=False)["elapsed_s"].diff().to_numpy(dtype=float)
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if len(deltas) == 0:
        raise ValueError("cannot estimate sampling rate without positive timestamp deltas")
    return float(1.0 / np.median(deltas))


def corrupt_samples(
    samples: pd.DataFrame,
    corruption: str,
    level: float | int,
    *,
    seed: int = 42,
    original_rate_hz: float | None = None,
) -> pd.DataFrame:
    """Corrupt development samples without changing labels, groups, or timestamps."""
    validate_canonical_samples(samples)
    if corruption not in {"noise", "dropout", "sampling_rate"}:
        raise ValueError(f"unsupported corruption: {corruption}")
    result = samples.copy(deep=True)
    rng = np.random.default_rng(seed)
    if corruption == "noise":
        sigma = float(level)
        if sigma < 0:
            raise ValueError("noise sigma must be non-negative")
        result.loc[:, ["mx", "my", "mz"]] += rng.normal(0.0, sigma, size=(len(result), 3))
        return result
    if corruption == "dropout":
        fraction = float(level)
        if not 0 <= fraction < 1:
            raise ValueError("dropout fraction must be in [0, 1)")
        if fraction == 0:
            return result
        keep = rng.random(len(result)) >= fraction
        for _, indices in result.groupby(["trajectory_id", "segment_index"], sort=False).groups.items():
            segment_indices = np.asarray(indices)
            if keep[segment_indices].sum() < 2:
                keep[segment_indices[:2]] = True
        dropped = result.loc[keep].copy()
        rate = estimate_sampling_rate_hz(samples) if original_rate_hz is None else float(original_rate_hz)
        return resample_segments(dropped, rate)
    if level == "original":
        return result
    target_rate = float(level)
    if target_rate <= 0:
        raise ValueError("sampling rate must be positive")
    rate = estimate_sampling_rate_hz(samples) if original_rate_hz is None else float(original_rate_hz)
    if target_rate >= rate * (1 - 1e-9):
        return result
    return resample_segments(result, target_rate)


def validate_corruption_invariants(original: pd.DataFrame, corrupted: pd.DataFrame) -> None:
    """Assert that corruption preserves trajectory-level metadata and sensor schema."""
    validate_canonical_samples(corrupted)
    for column in ("trajectory_id", "group_id", "label"):
        if set(original[column].astype(str)) != set(corrupted[column].astype(str)):
            raise AssertionError(f"corruption changed metadata domain for {column}")
    if not set(corrupted["trajectory_id"].astype(str)).issubset(set(original["trajectory_id"].astype(str))):
        raise AssertionError("corruption introduced a trajectory")
    if not set(corrupted["label"].astype(str)).issubset(set(original["label"].astype(str))):
        raise AssertionError("corruption introduced a label")
    if not set(corrupted["group_id"].astype(str)).issubset(set(original["group_id"].astype(str))):
        raise AssertionError("corruption introduced a group")
    if set(SENSOR_COLUMNS).intersection(original.columns) != set(SENSOR_COLUMNS).intersection(corrupted.columns):
        raise AssertionError("corruption changed sensor columns")