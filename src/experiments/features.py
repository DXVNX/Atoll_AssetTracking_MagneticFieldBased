"""Batch adapter from canonical windows to versioned feature-set contracts."""

from __future__ import annotations

import pandas as pd

from src.feature_sets import extract_window_features, get_feature_set


def extract_features_for_windows(samples: pd.DataFrame, windows: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    definition = get_feature_set(feature_set)
    by_trajectory = {key: group.set_index("sample_idx") for key, group in samples.groupby("trajectory_id")}
    records = []
    for row in windows.itertuples(index=False):
        segment = by_trajectory[row.trajectory_id].loc[row.first_sample_idx:row.last_sample_idx]
        values = extract_window_features(segment, feature_set)
        records.append({"window_id": row.window_id, **values.to_dict()})
    return pd.DataFrame(records)

