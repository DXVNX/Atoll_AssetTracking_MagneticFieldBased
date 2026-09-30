import numpy as np
import pandas as pd

from src.experiments.sequence import SequenceConfig, build_sequence_examples, fit_channel_normalizer, normalize_examples


def _frame():
    rows = []
    for trajectory in ("t1", "t2"):
        for segment in (0, 1):
            for index in range(40):
                sample_idx = segment * 40 + index
                rows.append({
                    "dataset_id": "ujiindoorloc_mag", "trajectory_id": trajectory, "group_id": trajectory,
                    "sample_idx": sample_idx, "timestamp": sample_idx * 100, "elapsed_s": sample_idx / 10,
                    "label": f"label_{segment}", "segment_index": segment, "source_split": "train_public",
                    "mx": float(index + (100 if trajectory == "t2" else 0)), "my": float(index + 1 + (100 if trajectory == "t2" else 0)), "mz": float(index + 2 + (100 if trajectory == "t2" else 0)),
                    "ax": 1.0, "ay": 2.0, "az": 3.0,
                })
    samples = pd.DataFrame(rows)
    windows = pd.DataFrame([{
        "window_id": "w1", "trajectory_id": "t1", "group_id": "t1", "segment_index": 0,
        "label": "label_0", "window_start_s": 0.0, "window_end_s": 2.0,
    }, {
        "window_id": "w2", "trajectory_id": "t2", "group_id": "t2", "segment_index": 1,
        "label": "label_1", "window_start_s": 4.0, "window_end_s": 6.0,
    }])
    return samples, windows


def test_sequence_shapes_labels_and_groups_are_deterministic():
    samples, windows = _frame()
    config = SequenceConfig(rate_hz=10, length_s=2, channels="magnetometer_accelerometer")
    first = build_sequence_examples(samples, windows, config)
    second = build_sequence_examples(samples, windows, config)
    assert [example.values.shape for example in first] == [(6, 20), (6, 20)]
    assert [example.label_index for example in first] == [0, 1]
    assert [example.trajectory_id for example in first] == ["t1", "t2"]
    np.testing.assert_array_equal(first[0].values, second[0].values)


def test_train_only_normalizer_does_not_use_validation_values():
    samples, windows = _frame()
    examples = build_sequence_examples(samples, windows, SequenceConfig())
    normalizer = fit_channel_normalizer(examples[:1])
    normalized = normalize_examples(examples, normalizer)
    assert np.allclose(normalized[0].values.mean(axis=1), 0.0, atol=1e-6)
    assert not np.allclose(normalized[1].values.mean(axis=1), 0.0, atol=1e-6)