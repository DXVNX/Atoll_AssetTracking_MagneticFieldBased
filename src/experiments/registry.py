"""Rebuild the result registry atomically from independent run artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

REGISTRY_COLUMNS = (
    "experiment_id", "run_id", "status", "dataset", "target", "feature_set",
    "window_s", "stride_s", "model", "hyperparameters_json", "cv_strategy",
    "accuracy_mean", "accuracy_std", "balanced_accuracy_mean", "macro_f1_mean",
    "macro_f1_std", "weighted_f1_mean", "top2_accuracy_mean", "fit_time_s",
    "inference_ms_per_window", "model_size_kb", "seed", "git_commit", "run_dir", "notes",
)


def rebuild_registry(results_dir: str | Path) -> pd.DataFrame:
    results_dir = Path(results_dir)
    records = []
    for metadata_path in sorted((results_dir / "runs").glob("*/*/run_metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config_path = metadata_path.parent / "config_resolved.yaml"
        import yaml
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        metrics = metadata.get("metrics_summary", {})
        records.append({
            "experiment_id": metadata.get("experiment_id"), "run_id": metadata.get("run_id"), "status": metadata.get("status"),
            "dataset": metadata.get("dataset"), "target": metadata.get("target"), "feature_set": config.get("features", {}).get("name"),
            "window_s": config.get("window", {}).get("length_s"), "stride_s": config.get("window", {}).get("stride_s"),
            "model": metadata.get("model"), "hyperparameters_json": json.dumps(metadata.get("hyperparameters", {}), sort_keys=True),
            "cv_strategy": metadata.get("split_strategy"), "accuracy_mean": metrics.get("accuracy_mean"), "accuracy_std": metrics.get("accuracy_std"),
            "balanced_accuracy_mean": metrics.get("balanced_accuracy_mean"), "macro_f1_mean": metrics.get("macro_f1_mean"), "macro_f1_std": metrics.get("macro_f1_std"),
            "weighted_f1_mean": metrics.get("weighted_f1_mean"), "top2_accuracy_mean": metrics.get("top2_accuracy_mean"),
            "fit_time_s": metrics.get("fit_time_s"), "inference_ms_per_window": metrics.get("inference_ms_per_window"), "model_size_kb": metrics.get("model_size_kb"),
            "seed": metadata.get("random_seed"), "git_commit": metadata.get("git_commit"),
            "run_dir": metadata_path.parent.relative_to(results_dir.parent).as_posix(), "notes": config.get("notes", ""),
        })
    frame = pd.DataFrame(records, columns=REGISTRY_COLUMNS)
    results_dir.mkdir(parents=True, exist_ok=True)
    temporary = results_dir / ".experiment_registry.csv.tmp"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, results_dir / "experiment_registry.csv")
    return frame

