"""Label support analysis based only on independent trajectories and splits."""

from __future__ import annotations

import pandas as pd


def build_label_support(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    cv_folds: int = 5,
    development_windows: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    records = []
    labels = sorted(set(train["label"]) | set(test["label"]))
    for label in labels:
        train_rows = train[train["label"] == label]
        test_rows = test[test["label"] == label]
        window_groups = 0
        if development_windows is not None:
            window_groups = development_windows.loc[development_windows["label"] == label, "group_id"].nunique()
        records.append({
            "segment_id": label,
            "train_trajectory_count": train_rows["trajectory_id"].nunique(),
            "test_trajectory_count": test_rows["trajectory_id"].nunique(),
            "train_sample_count": len(train_rows),
            "test_sample_count": len(test_rows),
            "development_window_trajectory_count": window_groups,
        })
    table = pd.DataFrame(records)
    train_labels = set(table.loc[table["train_trajectory_count"] > 0, "segment_id"])
    support_column = "development_window_trajectory_count" if development_windows is not None else "train_trajectory_count"
    eligible = set(table.loc[table[support_column] >= cv_folds, "segment_id"])
    test_labels = set(table.loc[table["test_trajectory_count"] > 0, "segment_id"])
    scopes = {
        "all_observed_train": sorted(train_labels),
        "cv_eligible": sorted(eligible),
        "rare_labels": sorted(train_labels - eligible),
        "unseen_test_labels": sorted(test_labels - train_labels),
    }
    return table, scopes

