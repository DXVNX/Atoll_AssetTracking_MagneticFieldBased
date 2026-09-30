import numpy as np
import pandas as pd

from src.experiments.robustness import corrupt_samples, estimate_sampling_rate_hz


def _samples():
    rows = []
    for trajectory in ("t1", "t2"):
        for segment in (0, 1):
            for index in range(10):
                sample_idx = segment * 10 + index
                rows.append({
                    "dataset_id": "ujiindoorloc_mag", "trajectory_id": trajectory,
                    "group_id": trajectory, "sample_idx": sample_idx,
                    "timestamp": sample_idx * 100, "elapsed_s": sample_idx / 10,
                    "label": f"s{segment}", "mx": 1.0, "my": 2.0, "mz": 3.0,
                    "ax": 0.0, "ay": 0.0, "az": 0.0, "segment_index": segment,
                    "source_split": "train_public",
                })
    return pd.DataFrame(rows)


def test_noise_is_deterministic_and_preserves_metadata():
    original = _samples()
    first = corrupt_samples(original, "noise", 1.0, seed=7)
    second = corrupt_samples(original, "noise", 1.0, seed=7)
    pd.testing.assert_frame_equal(first, second)
    for column in ("trajectory_id", "group_id", "label", "timestamp", "elapsed_s"):
        assert first[column].tolist() == original[column].tolist()
    assert not np.array_equal(first[["mx", "my", "mz"]], original[["mx", "my", "mz"]])


def test_dropout_and_sampling_rate_remain_segment_pure():
    original = _samples()
    dropout = corrupt_samples(original, "dropout", 0.3, seed=7, original_rate_hz=10)
    downsampled = corrupt_samples(original, "sampling_rate", 2, seed=7, original_rate_hz=10)
    assert set(dropout["trajectory_id"]) == {"t1", "t2"}
    assert set(downsampled["trajectory_id"]) == {"t1", "t2"}
    assert dropout.groupby(["trajectory_id", "segment_index"])["label"].nunique().max() == 1
    assert downsampled.groupby(["trajectory_id", "segment_index"])["label"].nunique().max() == 1
    assert np.isclose(estimate_sampling_rate_hz(original), 10.0)