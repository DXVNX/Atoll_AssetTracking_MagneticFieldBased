"""Canonical sensor schema independent of any raw dataset layout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "dataset_id", "trajectory_id", "group_id", "sample_idx", "timestamp",
    "elapsed_s", "label", "mx", "my", "mz",
)
OPTIONAL_COLUMNS = (
    "ax", "ay", "az", "orientation_azimuth", "orientation_pitch",
    "orientation_roll", "source_split", "segment_index", "directional_label",
    "device_id", "subject_id",
)
SENSOR_COLUMNS = (
    "mx", "my", "mz", "ax", "ay", "az", "orientation_azimuth",
    "orientation_pitch", "orientation_roll",
)


class SchemaValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SchemaValidationResult:
    n_rows: int
    n_trajectories: int
    n_groups: int
    warnings: tuple[str, ...] = ()


def adapt_uji_samples(frame: pd.DataFrame, dataset_id: str = "ujiindoorloc_mag") -> pd.DataFrame:
    """Map Phase 3 UJI output to the dataset-neutral canonical schema."""
    result = frame.rename(columns={
        "segment_id": "label",
        "directional_segment_id": "directional_label",
        "orientation_x": "orientation_azimuth",
        "orientation_y": "orientation_pitch",
        "orientation_z": "orientation_roll",
    }).copy()
    result.insert(0, "dataset_id", dataset_id)
    result.insert(2, "group_id", result["trajectory_id"])
    ordered = [c for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS if c in result.columns]
    extras = [c for c in result.columns if c not in ordered]
    return result.loc[:, ordered + extras]


def validate_canonical_samples(frame: pd.DataFrame) -> SchemaValidationResult:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SchemaValidationError(f"missing required columns: {missing}")
    if frame.empty:
        raise SchemaValidationError("canonical sample table is empty")
    for column in ("dataset_id", "trajectory_id", "group_id", "label"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise SchemaValidationError(f"{column} contains missing or empty values")
    numeric = ["sample_idx", "timestamp", "elapsed_s", "mx", "my", "mz"]
    numeric += [column for column in SENSOR_COLUMNS if column in frame.columns and column not in numeric]
    for column in numeric:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise SchemaValidationError(f"{column} must be numeric")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise SchemaValidationError(f"{column} contains non-finite values")
    keys = frame[["trajectory_id", "sample_idx"]]
    if keys.duplicated().any():
        raise SchemaValidationError("trajectory_id/sample_idx pairs must be unique")
    for trajectory_id, group in frame.groupby("trajectory_id", sort=False):
        if not group["sample_idx"].is_monotonic_increasing:
            raise SchemaValidationError(f"sample_idx is not monotonic in trajectory {trajectory_id}")
        if not group["timestamp"].is_monotonic_increasing:
            raise SchemaValidationError(f"timestamp is not monotonic in trajectory {trajectory_id}")
        if not group["elapsed_s"].is_monotonic_increasing:
            raise SchemaValidationError(f"elapsed_s is not monotonic in trajectory {trajectory_id}")
        if group["group_id"].nunique() != 1:
            raise SchemaValidationError(f"trajectory {trajectory_id} maps to multiple groups")
    return SchemaValidationResult(len(frame), frame["trajectory_id"].nunique(), frame["group_id"].nunique())

