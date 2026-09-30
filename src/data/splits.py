"""Group-aware split construction and hard leakage assertions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def assert_no_group_overlap(partitions: Mapping[str, Iterable[object]]) -> None:
    seen: dict[object, str] = {}
    for name, values in partitions.items():
        for group in set(values):
            if group in seen:
                raise ValueError(f"group overlap: {group!r} occurs in {seen[group]!r} and {name!r}")
            seen[group] = name


def make_group_cv_splits(windows: pd.DataFrame, n_splits: int = 5, *, stratified: bool = True) -> list[tuple[np.ndarray, np.ndarray]]:
    required = {"label", "group_id"}
    if missing := sorted(required - set(windows.columns)):
        raise ValueError(f"split input missing columns: {missing}")
    if windows["group_id"].nunique() < n_splits:
        raise ValueError("fewer groups than requested folds")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42) if stratified else GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(windows, windows["label"], groups=windows["group_id"]))
    for train_idx, validation_idx in splits:
        assert_no_group_overlap({
            "train": windows.iloc[train_idx]["group_id"],
            "validation": windows.iloc[validation_idx]["group_id"],
        })
    return splits


def largest_complete_stratified_group_cv(
    windows: pd.DataFrame, *, maximum_splits: int = 5, minimum_splits: int = 2
) -> tuple[int, list[tuple[np.ndarray, np.ndarray]], dict[int, list[list[str]]]]:
    """Choose the largest fold count whose every validation fold covers all labels."""
    labels = set(windows["label"])
    evidence: dict[int, list[list[str]]] = {}
    for n_splits in range(maximum_splits, minimum_splits - 1, -1):
        splits = make_group_cv_splits(windows, n_splits=n_splits)
        missing = [sorted(labels - set(windows.iloc[validation]["label"])) for _, validation in splits]
        evidence[n_splits] = missing
        if all(not fold_missing for fold_missing in missing):
            return n_splits, splits, evidence
    raise ValueError("no requested grouped CV configuration gives every validation fold full label coverage")

