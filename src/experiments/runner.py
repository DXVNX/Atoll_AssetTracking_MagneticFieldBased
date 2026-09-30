"""Unified grouped-CV experiment execution and artifact writing."""

from __future__ import annotations

import copy
import json
import os
import time
import traceback
import warnings
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline

from src.data.loading import load_uji_trajectories
from src.data.splits import assert_no_group_overlap

from .baselines import build_baseline
from .config import config_hash, validate_config
from .features import extract_features_for_windows
from src.feature_sets import get_feature_set
from .metrics import classification_metrics, trajectory_balanced_metrics
from .reproducibility import environment_snapshot, git_commit, set_global_seed, sha256_file


@lru_cache(maxsize=1)
def _environment_snapshot_cached() -> str:
    """Dependency state is invariant during one grid-run process."""
    return environment_snapshot()


def _mean_std(folds: list[dict], key: str) -> tuple[float, float]:
    values = [fold[key] for fold in folds]
    return float(np.mean(values)), float(np.std(values))


def run_experiment(config: dict, *, root: str | Path, smoke: bool = False, overwrite: bool = False, final_test: bool = False, seed: int | None = None) -> Path:
    root = Path(root).resolve()
    resolved = validate_config(copy.deepcopy(config))
    if seed is not None:
        resolved["seed"] = int(seed)
    seed = int(resolved["seed"])
    set_global_seed(seed)
    digest = config_hash(resolved)
    mode = "smoke" if smoke else "development"
    run_id = f"{mode}_s{seed}_{digest[:12]}"
    run_dir = root / resolved["outputs"]["root"] / "runs" / resolved["experiment_id"] / run_id
    if run_dir.exists() and not overwrite:
        raise FileExistsError(f"run already exists: {run_dir}; pass --overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    metadata = {
        "experiment_id": resolved["experiment_id"], "run_id": run_id,
        "status": "SMOKE" if smoke else "RUNNING", "timestamp": started,
        "git_commit": git_commit(root), "config_hash": digest, "random_seed": seed,
        "dataset": resolved["dataset"]["id"], "target": resolved["target"]["column"],
        "feature_schema_version": resolved["features"]["schema_version"],
        "model": resolved["model"]["name"], "hyperparameters": resolved["model"].get("parameters", {}),
        "split_strategy": resolved["evaluation"]["strategy"], "group_column": resolved["evaluation"]["group_column"],
        "environment": {"python": os.sys.version}, "warnings": [], "error": None,
    }
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    (run_dir / "environment.txt").write_text(_environment_snapshot_cached(), encoding="utf-8")
    log_lines = [f"{started} start {run_id} mode={mode}"]
    try:
        if final_test:
            raise ValueError("final holdout execution is intentionally unavailable until configurations are frozen")
        dataset_dir = (root / resolved["dataset"]["path"]).resolve()
        window_path = dataset_dir / "processed" / resolved["dataset"].get("window_manifest", "window_manifest_2s_stride_1s.csv")
        split_path = root / resolved["evaluation"]["split_manifest"]
        scope_path = dataset_dir / "manifests" / resolved["dataset"].get("label_scopes", "label_scopes.json")
        metadata["dataset_manifest_hash"] = sha256_file(dataset_dir / "manifests" / "raw_manifest.csv")
        windows = pd.read_csv(window_path)
        scopes = json.loads(scope_path.read_text(encoding="utf-8"))
        windows = windows[windows["label"].isin(scopes["cv_eligible"])].reset_index(drop=True)
        folds = pd.read_csv(split_path)
        fold_ids = sorted(folds["fold"].unique())[:1] if smoke else sorted(folds["fold"].unique())
        if smoke:
            smoke_fold = fold_ids[0]
            smoke_train = sorted(folds[(folds["fold"] == smoke_fold) & (folds["partition"] == "train")]["group_id"].astype(str).unique())[:20]
            smoke_validation = sorted(folds[(folds["fold"] == smoke_fold) & (folds["partition"] == "validation")]["group_id"].astype(str).unique())[:10]
            windows = windows[windows["group_id"].astype(str).isin(smoke_train + smoke_validation)].reset_index(drop=True)
        feature_definition = get_feature_set(resolved["features"]["name"])
        if resolved["features"]["schema_version"] != feature_definition.schema_version:
            raise ValueError(f"feature schema mismatch: config has {resolved['features']['schema_version']}, implementation has {feature_definition.schema_version}")
        if feature_definition.uji_status in {"DISABLED", "CUSTOM_ONLY"}:
            raise ValueError(f"feature set {feature_definition.name} is not valid for UJI ({feature_definition.uji_status})")
        cache_dir = root / resolved["outputs"]["root"] / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = config_hash({
            "feature_schema": feature_definition.schema_version,
            "window_manifest_sha256": sha256_file(window_path),
            "window_ids": windows["window_id"].astype(str).tolist(),
        })[:16]
        cache_path = cache_dir / f"window_features_{feature_definition.name}_{cache_key}.csv"
        if cache_path.exists():
            features = pd.read_csv(cache_path)
        else:
            samples = load_uji_trajectories(dataset_dir)
            features = extract_features_for_windows(samples, windows, feature_definition.name)
            features.to_csv(cache_path, index=False)
        features = features.set_index("window_id")
        feature_names = list(feature_definition.feature_names)
        fold_metrics, prediction_parts, split_artifact = [], [], {"strategy": resolved["evaluation"]["strategy"], "group_column": resolved["evaluation"]["group_column"], "folds": []}
        fit_total = 0.0
        inference_total = 0.0
        inference_count = 0
        final_model = None
        for fold_id in fold_ids:
            train_groups = sorted(folds[(folds["fold"] == fold_id) & (folds["partition"] == "train")]["group_id"].astype(str).unique())
            validation_groups = sorted(folds[(folds["fold"] == fold_id) & (folds["partition"] == "validation")]["group_id"].astype(str).unique())
            if smoke:
                train_groups, validation_groups = train_groups[:20], validation_groups[:10]
            assert_no_group_overlap({"train": train_groups, "validation": validation_groups})
            train_rows = windows[windows["group_id"].astype(str).isin(train_groups)]
            validation_rows = windows[windows["group_id"].astype(str).isin(validation_groups)]
            if train_rows.empty or validation_rows.empty:
                raise ValueError(f"fold {fold_id} has an empty partition")
            X_train = features.loc[train_rows["window_id"], feature_names].to_numpy()
            X_validation = features.loc[validation_rows["window_id"], feature_names].to_numpy()
            scaler_name = resolved["preprocessing"].get("scaler", "none")
            if scaler_name not in {"none", "standard", "robust"}:
                raise ValueError("preprocessing.scaler must be none, standard, or robust")
            model = build_baseline(resolved["model"]["name"], resolved["model"].get("parameters", {}), seed + int(fold_id))
            if scaler_name != "none":
                scaler = StandardScaler() if scaler_name == "standard" else RobustScaler()
                model = Pipeline([("scaler", scaler), ("model", model)])
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                start = time.perf_counter(); model.fit(X_train, train_rows["label"]); fit_s = time.perf_counter() - start
            for warning in caught_warnings:
                message = f"fold {fold_id}: {warning.category.__name__}: {warning.message}"
                if message not in metadata["warnings"]:
                    metadata["warnings"].append(message)
            start = time.perf_counter(); predicted = model.predict(X_validation); inference_s = time.perf_counter() - start
            scores = model.predict_proba(X_validation) if hasattr(model, "predict_proba") else None
            score_labels = getattr(model, "classes_", None)
            if scores is not None and not set(validation_rows["label"]).issubset(set(score_labels)):
                scores, score_labels = None, None
            metrics = classification_metrics(validation_rows["label"], predicted, scores=scores, score_labels=score_labels)
            pred = pd.DataFrame({
                "experiment_id": resolved["experiment_id"], "run_id": run_id, "fold": fold_id,
                "window_id": validation_rows["window_id"].to_numpy(), "trajectory_id": validation_rows["trajectory_id"].to_numpy(),
                "true_label": validation_rows["label"].to_numpy(), "predicted_label": predicted,
                "correct": validation_rows["label"].to_numpy() == predicted,
            })
            metrics.update(trajectory_balanced_metrics(pred))
            metrics.update({"fold": int(fold_id), "fit_time_s": fit_s, "inference_ms_per_window": inference_s * 1000 / len(pred)})
            fold_metrics.append(metrics); prediction_parts.append(pred)
            split_artifact["folds"].append({"fold": int(fold_id), "train_trajectory_ids": train_groups, "validation_trajectory_ids": validation_groups})
            fit_total += fit_s; inference_total += inference_s; inference_count += len(pred); final_model = model
        predictions = pd.concat(prediction_parts, ignore_index=True)
        summary = {}
        for key in ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
            summary[f"{key}_mean"], summary[f"{key}_std"] = _mean_std(fold_metrics, key)
        summary["trajectory_balanced_accuracy_mean"], summary["trajectory_balanced_accuracy_std"] = _mean_std(fold_metrics, "trajectory_accuracy_mean")
        summary["trajectory_balanced_macro_f1_mean"], summary["trajectory_balanced_macro_f1_std"] = _mean_std(fold_metrics, "trajectory_macro_f1_mean")
        if all("top2_accuracy" in fold for fold in fold_metrics):
            summary["top2_accuracy_mean"], summary["top2_accuracy_std"] = _mean_std(fold_metrics, "top2_accuracy")
        summary.update({"fit_time_s": fit_total, "inference_ms_per_window": inference_total * 1000 / inference_count, "n_folds": len(fold_metrics), "n_predictions": len(predictions)})
        joblib.dump(final_model, run_dir / "model.joblib")
        summary["model_size_kb"] = (run_dir / "model.joblib").stat().st_size / 1024
        (run_dir / "metrics.json").write_text(json.dumps({"summary": summary, "folds": fold_metrics}, indent=2) + "\n", encoding="utf-8")
        pd.DataFrame([{k: v for k, v in fold.items() if not isinstance(v, (dict, list))} for fold in fold_metrics]).to_csv(run_dir / "fold_metrics.csv", index=False)
        predictions.to_csv(run_dir / "predictions.csv", index=False)
        (run_dir / "splits.json").write_text(json.dumps(split_artifact, indent=2) + "\n", encoding="utf-8")
        metadata["status"] = "SMOKE" if smoke else "COMPLETED"
        metadata["metrics_summary"] = summary
        log_lines.append(f"{datetime.now(timezone.utc).isoformat()} complete predictions={len(predictions)}")
    except Exception as exc:
        metadata["status"] = "FAILED"
        metadata["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        log_lines.append(f"{datetime.now(timezone.utc).isoformat()} failed {type(exc).__name__}: {exc}")
        raise
    finally:
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (run_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return run_dir

