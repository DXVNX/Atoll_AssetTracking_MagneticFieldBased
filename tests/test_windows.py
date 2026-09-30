import pandas as pd
import pytest

from src.data.schemas import adapt_uji_samples
from src.data.windows import WindowConfig, generate_windows, resample_segments


def _trajectory(trajectory="t1", labels=None, times=None):
    times = times or [i / 10 for i in range(41)]
    labels = labels or ["a"] * len(times)
    segments = [0 if label == labels[0] else 1 for label in labels]
    raw = pd.DataFrame({
        "trajectory_id": trajectory, "sample_idx": range(len(times)),
        "timestamp": [1000 + round(t * 1000) for t in times], "elapsed_s": times,
        "segment_id": labels, "directional_segment_id": [f"d_{x}" for x in labels],
        "segment_index": segments, "source_split": "train_public",
        "mx": range(len(times)), "my": range(len(times)), "mz": range(len(times)),
    })
    return adapt_uji_samples(raw)


def test_windows_are_deterministic_and_duration_is_timestamp_based():
    frame = _trajectory()
    first = generate_windows(frame, WindowConfig(2.0, 1.0))
    second = generate_windows(frame, WindowConfig(2.0, 1.0))
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 3
    assert (first["window_end_s"] - first["window_start_s"] == 2.0).all()
    assert first["n_samples"].tolist() == [20, 20, 20]


def test_windows_never_cross_trajectory_or_segment_boundaries():
    left = _trajectory("left", ["a"] * 16 + ["b"] * 25)
    right = _trajectory("right")
    windows = generate_windows(pd.concat([left, right], ignore_index=True), WindowConfig(2.0, 1.0))
    assert set(windows["trajectory_id"]) == {"left", "right"}
    assert (windows["group_id"] == windows["trajectory_id"]).all()
    left_windows = windows[windows["trajectory_id"] == "left"]
    assert set(left_windows["label"]) == {"b"}
    assert (left_windows["first_sample_idx"] >= 16).all()


def test_supported_window_lengths():
    frame = _trajectory(times=[i / 10 for i in range(61)])
    for length in (0.5, 1.0, 2.0, 3.0, 5.0):
        windows = generate_windows(frame, WindowConfig(length, 1.0))
        assert not windows.empty


def test_resampling_stays_inside_segments():
    frame = _trajectory(labels=["a"] * 21 + ["b"] * 20)
    result = resample_segments(frame, 5.0)
    assert result.groupby("segment_index")["label"].nunique().max() == 1
    assert not result[["trajectory_id", "sample_idx"]].duplicated().any()
    assert set(result["label"]) == {"a", "b"}


def test_invalid_window_config():
    with pytest.raises(ValueError):
        WindowConfig(0, 1)

