"""Versioned magnetic and inertial window feature sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.adapters.uji_orientation import add_uji_world_frame
from src.features import FEATURE_COLS_V1, FEATURE_COLS_V2, extract_features, extract_features_v2

STAT_NAMES = ("mean", "std", "median", "min", "max", "p25", "p75", "range", "iqr")


@dataclass(frozen=True)
class FeatureSet:
    name: str
    version: str
    feature_names: tuple[str, ...]
    extractor: Callable[[pd.DataFrame], np.ndarray]
    uji_status: str = "ENABLED"

    @property
    def schema_version(self) -> str:
        return f"{self.name}@{self.version}"


def _stats(series: pd.Series, prefix: str) -> tuple[list[str], list[float]]:
    values = series.to_numpy(dtype=float)
    q25, q75 = np.quantile(values, [0.25, 0.75])
    names = [f"{prefix}_{name}" for name in STAT_NAMES]
    result = [np.mean(values), np.std(values, ddof=1), np.median(values), np.min(values), np.max(values), q25, q75, np.max(values) - np.min(values), q75 - q25]
    return names, result


def _magnitude(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(np.linalg.norm(frame[["mx", "my", "mz"]].to_numpy(dtype=float), axis=1), index=frame.index)


def _combined(frame: pd.DataFrame, columns: tuple[str, ...], include_magnitude: bool = False) -> np.ndarray:
    values: list[float] = []
    if include_magnitude:
        values.extend(_stats(_magnitude(frame), "mag")[1])
    for column in columns:
        values.extend(_stats(frame[column], column)[1])
    return np.asarray(values, dtype=float)


def _legacy_v1(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray([extract_features(frame)[name] for name in FEATURE_COLS_V1], dtype=float)


def _legacy_v2(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray([extract_features_v2(frame)[name] for name in FEATURE_COLS_V2], dtype=float)


def _uji_legacy_v2(frame: pd.DataFrame) -> np.ndarray:
    world = add_uji_world_frame(frame)
    adapted = pd.DataFrame({
        "magnitude_smooth": _magnitude(world),
        "mh": world["mw_horizontal"], "mv": world["mw_vertical"],
        "Bx_w": world["mw_east"], "By_w": world["mw_north"],
        "inclination": world["mw_inclination"],
    })
    return _legacy_v2(adapted)


def _expanded_world(frame: pd.DataFrame) -> np.ndarray:
    world = add_uji_world_frame(frame)
    return _combined(world, ("mw_east", "mw_north", "mw_up", "mw_horizontal", "mw_vertical", "mw_inclination"), include_magnitude=True)


MAG9 = tuple(f"mag_{name}" for name in STAT_NAMES)
BODY_NAMES = MAG9 + tuple(f"{axis}_{name}" for axis in ("mx", "my", "mz") for name in STAT_NAMES)
ACC_NAMES = tuple(f"{axis}_{name}" for axis in ("ax", "ay", "az") for name in STAT_NAMES)
WORLD_NAMES = MAG9 + tuple(f"{axis}_{name}" for axis in ("mw_east", "mw_north", "mw_up", "mw_horizontal", "mw_vertical", "mw_inclination") for name in STAT_NAMES)

FEATURE_SETS = {
    "mag_mean": FeatureSet("mag_mean", "1", ("mag_mean",), lambda f: np.array([_magnitude(f).mean()])),
    "magnitude_stats": FeatureSet("magnitude_stats", "1", MAG9, lambda f: _combined(f, (), include_magnitude=True)),
    "body_xyz_magnitude": FeatureSet("body_xyz_magnitude", "1", BODY_NAMES, lambda f: _combined(f, ("mx", "my", "mz"), include_magnitude=True)),
    "legacy_v1_mag5": FeatureSet("legacy_v1_mag5", "1", tuple(FEATURE_COLS_V1), _legacy_v1, uji_status="DISABLED"),
    "legacy_v2_world13": FeatureSet("legacy_v2_world13", "1", tuple(FEATURE_COLS_V2), _legacy_v2, uji_status="CUSTOM_ONLY"),
    "uji_legacy_v2_world13": FeatureSet("uji_legacy_v2_world13", "1", tuple(FEATURE_COLS_V2), _uji_legacy_v2, uji_status="PARTIAL"),
    "uji_expanded_world": FeatureSet("uji_expanded_world", "1", WORLD_NAMES, _expanded_world, uji_status="PARTIAL"),
    "accelerometer_stats": FeatureSet("accelerometer_stats", "1", ACC_NAMES, lambda f: _combined(f, ("ax", "ay", "az"))),
    "body_magnetic_accelerometer": FeatureSet("body_magnetic_accelerometer", "1", BODY_NAMES + ACC_NAMES, lambda f: np.concatenate((_combined(f, ("mx", "my", "mz"), True), _combined(f, ("ax", "ay", "az"))))),
    "magnitude_accelerometer": FeatureSet("magnitude_accelerometer", "1", MAG9 + ACC_NAMES, lambda f: np.concatenate((_combined(f, (), True), _combined(f, ("ax", "ay", "az"))))),
    "uji_world_accelerometer": FeatureSet("uji_world_accelerometer", "1", WORLD_NAMES + ACC_NAMES, lambda f: np.concatenate((_expanded_world(f), _combined(f, ("ax", "ay", "az")))), uji_status="PARTIAL"),
}


def get_feature_set(name: str) -> FeatureSet:
    try:
        return FEATURE_SETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown feature set {name!r}; available: {sorted(FEATURE_SETS)}") from exc


def extract_window_features(window_df: pd.DataFrame, feature_set: str, config: dict | None = None) -> pd.Series:
    definition = get_feature_set(feature_set)
    values = definition.extractor(window_df)
    if values.shape != (len(definition.feature_names),):
        raise ValueError(f"{definition.name} produced {values.shape}, expected {(len(definition.feature_names),)}")
    if not np.isfinite(values).all():
        raise ValueError(f"{definition.name} produced non-finite values")
    return pd.Series(values, index=definition.feature_names, dtype=float)

