import numpy as np
import pandas as pd
import pytest

from src.data.schemas import SchemaValidationError, adapt_uji_samples, validate_canonical_samples


def canonical_frame() -> pd.DataFrame:
    return adapt_uji_samples(pd.DataFrame({
        "trajectory_id": ["t1", "t1"], "sample_idx": [0, 1],
        "timestamp": [1000, 1100], "elapsed_s": [0.0, 0.1],
        "segment_id": ["s1", "s1"], "directional_segment_id": ["d1", "d1"],
        "segment_index": [0, 0], "source_split": ["train_public", "train_public"],
        "mx": [1.0, 2.0], "my": [2.0, 3.0], "mz": [3.0, 4.0],
    }))


def test_uji_adapter_and_schema_validation():
    frame = canonical_frame()
    result = validate_canonical_samples(frame)
    assert result.n_rows == 2
    assert frame["group_id"].tolist() == frame["trajectory_id"].tolist()
    assert frame["label"].tolist() == ["s1", "s1"]


@pytest.mark.parametrize("mutation,match", [
    (lambda f: f.drop(columns="mx"), "missing required"),
    (lambda f: f.assign(mx=[1.0, np.inf]), "non-finite"),
    (lambda f: f.assign(timestamp=[1100, 1000]), "timestamp is not monotonic"),
    (lambda f: pd.concat([f, f.iloc[[0]]]), "must be unique"),
    (lambda f: f.assign(label=["", "s1"]), "label contains"),
])
def test_schema_rejects_invalid_data(mutation, match):
    with pytest.raises(SchemaValidationError, match=match):
        validate_canonical_samples(mutation(canonical_frame()))

