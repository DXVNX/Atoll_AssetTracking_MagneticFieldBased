"""Timestamp-based, segment-pure deterministic windows and resampling."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schemas import SENSOR_COLUMNS, validate_canonical_samples

WINDOW_COLUMNS = (
    "window_id", "dataset_id", "trajectory_id", "group_id", "label",
    "source_split", "window_start_s", "window_end_s", "n_samples",
    "first_sample_idx", "last_sample_idx",
)


@dataclass(frozen=True)
class WindowConfig:
    length_s: float = 2.0
    stride_s: float = 1.0
    min_samples: int = 2

    def __post_init__(self) -> None:
        if self.length_s <= 0 or self.stride_s <= 0:
            raise ValueError("window length and stride must be positive")
        if self.min_samples < 1:
            raise ValueError("min_samples must be positive")


def _window_id(dataset: str, trajectory: str, segment: object, start: float, end: float) -> str:
    token = f"{dataset}|{trajectory}|{segment}|{start:.9f}|{end:.9f}"
    return "win_" + hashlib.sha256(token.encode("utf-8")).hexdigest()[:20]


def generate_windows(frame: pd.DataFrame, config: WindowConfig = WindowConfig()) -> pd.DataFrame:
    """Return metadata for half-open [start, end) windows wholly inside one segment."""
    validate_canonical_samples(frame)
    required = {"source_split", "segment_index"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"window generation requires columns: {missing}")
    records: list[dict] = []
    grouping = ["dataset_id", "trajectory_id", "segment_index"]
    for (dataset_id, trajectory_id, segment_index), segment in frame.groupby(grouping, sort=True):
        segment = segment.sort_values("elapsed_s")
        if segment["label"].nunique() != 1 or segment["group_id"].nunique() != 1:
            raise ValueError(f"segment {trajectory_id}/{segment_index} is not label/group pure")
        first_time = float(segment["elapsed_s"].iloc[0])
        last_time = float(segment["elapsed_s"].iloc[-1])
        start = first_time
        while start + config.length_s <= last_time + 1e-12:
            end = start + config.length_s
            samples = segment[(segment["elapsed_s"] >= start - 1e-12) & (segment["elapsed_s"] < end - 1e-12)]
            if len(samples) >= config.min_samples:
                records.append({
                    "window_id": _window_id(str(dataset_id), str(trajectory_id), segment_index, start, end),
                    "dataset_id": dataset_id,
                    "trajectory_id": trajectory_id,
                    "group_id": segment["group_id"].iloc[0],
                    "label": segment["label"].iloc[0],
                    "source_split": segment["source_split"].iloc[0],
                    "window_start_s": start,
                    "window_end_s": end,
                    "n_samples": len(samples),
                    "first_sample_idx": int(samples["sample_idx"].iloc[0]),
                    "last_sample_idx": int(samples["sample_idx"].iloc[-1]),
                })
            start = first_time + (round((start - first_time) / config.stride_s) + 1) * config.stride_s
    result = pd.DataFrame.from_records(records, columns=WINDOW_COLUMNS)
    if not result.empty and result["window_id"].duplicated().any():
        raise RuntimeError("deterministic window IDs are not unique")
    return result


def resample_segments(frame: pd.DataFrame, target_rate_hz: float) -> pd.DataFrame:
    """Linearly resample each trajectory segment independently."""
    validate_canonical_samples(frame)
    if target_rate_hz <= 0:
        raise ValueError("target_rate_hz must be positive")
    if "segment_index" not in frame:
        raise ValueError("resampling requires segment_index")
    pieces = []
    step = 1.0 / target_rate_hz
    for _, segment in frame.groupby(["dataset_id", "trajectory_id", "segment_index"], sort=True):
        segment = segment.sort_values("elapsed_s")
        if segment["label"].nunique() != 1:
            raise ValueError("cannot resample a segment containing multiple labels")
        old_t = segment["elapsed_s"].to_numpy(dtype=float)
        if len(old_t) < 2 or old_t[-1] <= old_t[0]:
            continue
        new_t = np.arange(old_t[0], old_t[-1] + step * 0.5, step)
        new_t = new_t[new_t <= old_t[-1] + 1e-12]
        base = {column: segment[column].iloc[0] for column in segment.columns if column not in SENSOR_COLUMNS and column not in {"sample_idx", "timestamp", "elapsed_s"}}
        output = pd.DataFrame({column: [value] * len(new_t) for column, value in base.items()})
        output["sample_idx"] = np.arange(len(new_t), dtype=int)
        output["elapsed_s"] = new_t
        output["timestamp"] = np.rint(float(segment["timestamp"].iloc[0]) + (new_t - old_t[0]) * 1000).astype("int64")
        for column in SENSOR_COLUMNS:
            if column in segment:
                output[column] = np.interp(new_t, old_t, segment[column].to_numpy(dtype=float))
        pieces.append(output)
    if not pieces:
        return pd.DataFrame(columns=frame.columns)
    result = pd.concat(pieces, ignore_index=True).sort_values(["trajectory_id", "elapsed_s", "segment_index"])
    result["sample_idx"] = result.groupby("trajectory_id", sort=False).cumcount()
    return result.reset_index(drop=True)

