import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from src.data.loading import FinalTestLockedError, load_uji_trajectories
from src.data.preprocessing import fit_transform_fold
from src.data.splits import assert_no_group_overlap, largest_complete_stratified_group_cv, make_group_cv_splits


def test_overlap_hard_fails_and_disjoint_passes():
    with pytest.raises(ValueError, match="group overlap"):
        assert_no_group_overlap({"train": ["t1", "t2"], "validation": ["t2", "t3"]})
    assert_no_group_overlap({"train": ["t1", "t2"], "validation": ["t3"]})


def test_group_cv_never_separates_a_trajectory():
    rows = []
    for group in range(10):
        for item in range(3):
            rows.append({"window_id": f"w{group}_{item}", "group_id": f"t{group}", "label": f"s{group % 2}"})
    windows = pd.DataFrame(rows)
    for train_idx, validation_idx in make_group_cv_splits(windows, 5):
        assert set(windows.iloc[train_idx]["group_id"]).isdisjoint(windows.iloc[validation_idx]["group_id"])


def test_largest_complete_group_cv_reports_feasibility():
    rows = []
    for group in range(6):
        rows.append({"group_id": f"t{group}", "label": "common"})
        if group < 2:
            rows.append({"group_id": f"t{group}", "label": "rare"})
    windows = pd.DataFrame(rows)
    with pytest.warns(UserWarning):
        count, splits, evidence = largest_complete_stratified_group_cv(windows, maximum_splits=3)
    assert count == 2
    assert len(splits) == 2
    assert any(evidence[3])


def test_official_test_is_excluded_by_default(tmp_path):
    with pytest.raises(FinalTestLockedError):
        load_uji_trajectories(tmp_path, source_split="test_public")


class RecordingTransformer(BaseEstimator, TransformerMixin):
    def fit(self, values, y=None):
        self.seen_ = np.asarray(values).copy()
        return self

    def transform(self, values):
        return np.asarray(values)


def test_transformer_is_fitted_only_on_fold_training_rows():
    values = np.arange(12).reshape(6, 2)
    result = fit_transform_fold(RecordingTransformer(), values, [0, 1, 2, 3], [4, 5])
    assert result.transformer.seen_.tolist() == values[:4].tolist()
    assert result.validation.tolist() == values[4:].tolist()

