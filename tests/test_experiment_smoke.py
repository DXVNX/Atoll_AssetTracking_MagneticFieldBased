import json
from pathlib import Path

import pandas as pd
import pytest

from src.experiments.config import config_hash
from src.experiments.registry import rebuild_registry
from src.experiments.runner import run_experiment


def _fixture(root: Path, overlap=False):
    dataset = root / "dataset"
    (dataset / "interim" / "train_public").mkdir(parents=True)
    (dataset / "processed").mkdir()
    (dataset / "manifests").mkdir()
    windows = []
    assignments = {"t1": "a", "t2": "b", "t3": "a", "t4": "b"}
    for trajectory, label in assignments.items():
        raw = pd.DataFrame({
            "sample_idx": [0, 1], "timestamp": [1000, 1100], "elapsed_s": [0.0, 0.1],
            "mx": [1.0, 2.0], "my": [1.0, 2.0], "mz": [1.0, 2.0],
            "trajectory_id": trajectory, "source_split": "train_public", "trajectory_type": "line",
            "segment_index": 0, "segment_id": label, "directional_segment_id": "d_" + label,
        })
        raw.to_csv(dataset / "interim" / "train_public" / f"{trajectory}.csv", index=False)
        windows.append({"window_id": "w_" + trajectory, "dataset_id": "ujiindoorloc_mag", "trajectory_id": trajectory, "group_id": trajectory, "label": label, "source_split": "train_public", "window_start_s": 0, "window_end_s": 2, "n_samples": 2, "first_sample_idx": 0, "last_sample_idx": 1})
    pd.DataFrame(windows).to_csv(dataset / "processed" / "windows.csv", index=False)
    (dataset / "manifests" / "raw_manifest.csv").write_text("trajectory_id\nt1\n", encoding="utf-8")
    (dataset / "manifests" / "scopes.json").write_text(json.dumps({"cv_eligible": ["a", "b"]}), encoding="utf-8")
    folds = [
        {"fold": 0, "partition": "train", "group_id": "t1"},
        {"fold": 0, "partition": "train", "group_id": "t2"},
        {"fold": 0, "partition": "validation", "group_id": "t1" if overlap else "t3"},
        {"fold": 0, "partition": "validation", "group_id": "t4"},
    ]
    pd.DataFrame(folds).to_csv(root / "folds.csv", index=False)
    return {
        "experiment_id": "E_FIXTURE", "name": "fixture", "dataset": {"id": "ujiindoorloc_mag", "path": "dataset", "window_manifest": "windows.csv", "label_scopes": "scopes.json"},
        "target": {"column": "label"}, "window": {"length_s": 2.0, "stride_s": 1.0}, "preprocessing": {"scaler": "none"},
        "features": {"name": "magnitude_stats", "schema_version": "magnitude_stats@1"},
        "evaluation": {"strategy": "fixture_group_fold", "split_manifest": "folds.csv", "group_column": "trajectory_id"},
        "model": {"name": "majority", "parameters": {}}, "outputs": {"root": "results"}, "seed": 42,
    }


def test_smoke_experiment_end_to_end_and_split_artifact(tmp_path):
    config = _fixture(tmp_path)
    run_dir = run_experiment(config, root=tmp_path, smoke=True)
    required = {"config_resolved.yaml", "metrics.json", "fold_metrics.csv", "predictions.csv", "splits.json", "environment.txt", "run.log", "run_metadata.json", "model.joblib"}
    assert required <= {path.name for path in run_dir.iterdir()}
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["status"] == "SMOKE"
    split = json.loads((run_dir / "splits.json").read_text())["folds"][0]
    assert set(split["train_trajectory_ids"]).isdisjoint(split["validation_trajectory_ids"])


def test_group_overlap_fails_and_writes_failed_metadata(tmp_path):
    config = _fixture(tmp_path, overlap=True)
    with pytest.raises(ValueError, match="group overlap"):
        run_experiment(config, root=tmp_path, smoke=True)
    metadata_path = next((tmp_path / "results" / "runs").glob("*/*/run_metadata.json"))
    metadata = json.loads(metadata_path.read_text())
    assert metadata["status"] == "FAILED"
    assert metadata["error"]["type"] == "ValueError"


def test_registry_rebuild_includes_smoke_but_marks_status(tmp_path):
    config = _fixture(tmp_path)
    run_experiment(config, root=tmp_path, smoke=True)
    registry = rebuild_registry(tmp_path / "results")
    assert len(registry) == 1
    assert registry.iloc[0]["status"] == "SMOKE"
    assert (tmp_path / "results" / "experiment_registry.csv").exists()

