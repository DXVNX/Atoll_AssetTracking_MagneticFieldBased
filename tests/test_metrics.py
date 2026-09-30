import pandas as pd
import pytest

from src.experiments.metrics import classification_metrics, trajectory_balanced_metrics


def test_classification_metrics_are_correct():
    metrics = classification_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["macro_precision"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert metrics["macro_recall"] == pytest.approx(0.75)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["per_class"]["a"]["support"] == 2


def test_trajectory_balancing_weights_trajectories_equally():
    predictions = pd.DataFrame({
        "trajectory_id": ["long"] * 9 + ["short"],
        "true_label": ["a"] * 10,
        "predicted_label": ["a"] * 9 + ["b"],
    })
    metrics = trajectory_balanced_metrics(predictions)
    assert metrics["trajectory_accuracy_mean"] == pytest.approx(0.5)

