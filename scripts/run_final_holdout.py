"""One-time explicit final holdout evaluation for frozen candidates."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from src.data.loading import load_uji_trajectories
from src.data.windows import WindowConfig, generate_windows
from src.experiments.baselines import build_baseline
from src.experiments.benchmark import benchmark_callable, serialized_size_kb
from src.experiments.deep import MLPClassifier, set_torch_seed
from src.experiments.features import extract_features_for_windows
from src.experiments.metrics import classification_metrics
from src.experiments.reproducibility import git_commit

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data/external/ujiindoorloc_mag"
SCOPE_PATH = DATASET_DIR / "manifests/label_scopes.json"
SEED = 42


def _per_trajectory(predictions: pd.DataFrame, known_labels: set[str]) -> pd.DataFrame:
    rows = []
    for trajectory_id, group in predictions.groupby("trajectory_id", sort=True):
        known = group[~group["is_unseen_label"]]
        rows.append({"trajectory_id": trajectory_id, "n_windows": len(group), "n_unseen_windows": int(group["is_unseen_label"].sum()), "known_window_count": len(known), "known_accuracy": float(known["correct"].mean()) if len(known) else np.nan})
    return pd.DataFrame(rows)


def _evaluate_classifier(experiment_id: str, config: dict, train_samples: pd.DataFrame, train_windows: pd.DataFrame, test_samples: pd.DataFrame, test_windows: pd.DataFrame, known_labels: list[str]) -> dict:
    feature_name = config["features"]["name"]
    train_features = extract_features_for_windows(train_samples, train_windows, feature_name).set_index("window_id")
    test_features = extract_features_for_windows(test_samples, test_windows, feature_name).set_index("window_id")
    model = build_baseline(config["model"]["name"], config["model"]["parameters"], SEED)
    model.fit(train_features.to_numpy(), train_windows.set_index("window_id").loc[train_features.index, "label"])
    predictions = model.predict(test_features.loc[test_windows["window_id"]].to_numpy())
    true_labels = test_windows.set_index("window_id").loc[test_features.index, "label"].to_numpy()
    known_mask = np.isin(true_labels, known_labels)
    prediction_frame = pd.DataFrame({"window_id": test_features.index, "trajectory_id": test_windows.set_index("window_id").loc[test_features.index, "trajectory_id"].to_numpy(), "true_label": true_labels, "predicted_label": predictions, "is_unseen_label": ~known_mask, "correct": np.where(known_mask, true_labels == predictions, np.nan)})
    closed = prediction_frame[~prediction_frame["is_unseen_label"]]
    metrics = classification_metrics(closed["true_label"], closed["predicted_label"], labels=known_labels) if len(closed) else {"accuracy": np.nan, "balanced_accuracy": np.nan, "macro_f1": np.nan, "weighted_f1": np.nan}
    run_dir = ROOT / "results/runs" / experiment_id / "final_s42"
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, run_dir / "model.joblib")
    prediction_frame.to_csv(run_dir / "predictions.csv", index=False)
    _per_trajectory(prediction_frame, set(known_labels)).to_csv(run_dir / "per_trajectory_metrics.csv", index=False)
    per_class = closed.groupby("true_label").apply(lambda rows: pd.Series({"support": len(rows), "accuracy": float(rows["correct"].mean())})).reset_index()
    per_class.to_csv(run_dir / "per_class_metrics.csv", index=False)
    confusion = pd.crosstab(closed["true_label"], closed["predicted_label"]).reindex(index=known_labels, columns=known_labels, fill_value=0)
    confusion.to_csv(run_dir / "confusion_matrix.csv")
    sample_features = test_features.iloc[[0]].to_numpy()
    inference_median, inference_p95 = benchmark_callable(lambda: model.predict(sample_features))
    summary = {"experiment_id": experiment_id, "model": config["model"]["name"], "feature_set": feature_name, "accuracy": metrics.get("accuracy"), "balanced_accuracy": metrics.get("balanced_accuracy"), "macro_f1": metrics.get("macro_f1"), "weighted_f1": metrics.get("weighted_f1"), "model_size_kb": serialized_size_kb(run_dir / "model.joblib"), "latency_median_ms": inference_median, "latency_p95_ms": inference_p95, "n_test_windows": len(prediction_frame), "n_closed_set_windows": int(known_mask.sum()), "n_unseen_windows": int((~known_mask).sum()), "unseen_label_count": int(prediction_frame.loc[~known_mask, "true_label"].nunique()), "status": "COMPLETED"}
    _write_metadata(run_dir, experiment_id, config, summary)
    return summary


def _fit_mlp(values: np.ndarray, labels: np.ndarray, n_classes: int, epochs: int = 47) -> MLPClassifier:
    set_torch_seed(SEED)
    model = MLPClassifier(values.shape[1], n_classes)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    weights = np.zeros(n_classes, dtype=np.float32); present = counts > 0; weights[present] = counts[present].sum() / (present.sum() * counts[present])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(weights))
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(values).float(), torch.from_numpy(labels).long()), batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    model.train()
    for _ in range(epochs):
        for batch_values, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True); loss_function(model(batch_values), batch_labels).backward(); optimizer.step()
    return model


def _evaluate_mlp(experiment_id: str, config: dict, train_samples: pd.DataFrame, train_windows: pd.DataFrame, test_samples: pd.DataFrame, test_windows: pd.DataFrame, known_labels: list[str]) -> dict:
    feature_name = config["features"]["name"]
    train_features = extract_features_for_windows(train_samples, train_windows, feature_name).set_index("window_id")
    test_features = extract_features_for_windows(test_samples, test_windows, feature_name).set_index("window_id")
    train_meta = train_windows.set_index("window_id").loc[train_features.index]; test_meta = test_windows.set_index("window_id").loc[test_features.index]
    label_to_index = {label: index for index, label in enumerate(known_labels)}
    train_labels = train_meta["label"].map(label_to_index).to_numpy(dtype=np.int64)
    scaler = StandardScaler().fit(train_features.to_numpy()); train_values = scaler.transform(train_features.to_numpy()).astype(np.float32); test_values = scaler.transform(test_features.to_numpy()).astype(np.float32)
    model = _fit_mlp(train_values, train_labels, len(known_labels), epochs=47)
    with torch.no_grad(): predictions = model(torch.from_numpy(test_values)).argmax(dim=1).numpy()
    true_labels = test_meta["label"].to_numpy(); known_mask = np.isin(true_labels, known_labels)
    prediction_frame = pd.DataFrame({"window_id": test_features.index, "trajectory_id": test_meta["trajectory_id"].to_numpy(), "true_label": true_labels, "predicted_label": [known_labels[index] for index in predictions], "is_unseen_label": ~known_mask, "correct": np.where(known_mask, true_labels == np.asarray([known_labels[index] for index in predictions]), np.nan)})
    closed = prediction_frame[~prediction_frame["is_unseen_label"]]; metrics = classification_metrics(closed["true_label"], closed["predicted_label"], labels=known_labels)
    run_dir = ROOT / "results/runs" / experiment_id / "final_s42"; run_dir.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), run_dir / "model.pt"); joblib.dump(scaler, run_dir / "scaler.joblib"); prediction_frame.to_csv(run_dir / "predictions.csv", index=False); _per_trajectory(prediction_frame, set(known_labels)).to_csv(run_dir / "per_trajectory_metrics.csv", index=False); closed.groupby("true_label").apply(lambda rows: pd.Series({"support": len(rows), "accuracy": float(rows["correct"].mean())})).reset_index().to_csv(run_dir / "per_class_metrics.csv", index=False); pd.crosstab(closed["true_label"], closed["predicted_label"]).reindex(index=known_labels, columns=known_labels, fill_value=0).to_csv(run_dir / "confusion_matrix.csv")
    inference_median, inference_p95 = benchmark_callable(lambda: model(torch.from_numpy(test_values[:1])))
    size_kb = (run_dir / "model.pt").stat().st_size / 1024 + (run_dir / "scaler.joblib").stat().st_size / 1024
    summary = {"experiment_id": experiment_id, "model": "mlp", "feature_set": feature_name, "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"], "macro_f1": metrics["macro_f1"], "weighted_f1": metrics["weighted_f1"], "model_size_kb": size_kb, "latency_median_ms": inference_median, "latency_p95_ms": inference_p95, "n_test_windows": len(prediction_frame), "n_closed_set_windows": int(known_mask.sum()), "n_unseen_windows": int((~known_mask).sum()), "unseen_label_count": int(prediction_frame.loc[~known_mask, "true_label"].nunique()), "status": "COMPLETED"}
    _write_metadata(run_dir, experiment_id, config, summary); return summary


def _write_metadata(run_dir: Path, experiment_id: str, config: dict, summary: dict) -> None:
    resolved = dict(config); resolved["mode"] = "FINAL"; resolved["final_test"] = True; resolved["development_frozen"] = True; (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    metadata = {"experiment_id": experiment_id, "run_id": "final_s42", "status": "FINAL_HOLDOUT", "mode": "FINAL", "final_test": True, "development_frozen": True, "timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(ROOT), "random_seed": SEED, "dataset": "ujiindoorloc_mag", "target": "label", "model": summary["model"], "feature_schema_version": config["features"].get("schema_version", config["features"].get("name")), "metrics_summary": summary, "public_test_access": "explicit_final_holdout_only"}
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    train_samples = load_uji_trajectories(DATASET_DIR, source_split="train_public")
    test_samples = load_uji_trajectories(DATASET_DIR, source_split="test_public", allow_final_test=True)
    labels = json.loads(SCOPE_PATH.read_text(encoding="utf-8"))["cv_eligible"]
    train_windows = pd.read_csv(DATASET_DIR / "processed/window_manifest_2s_stride_1s.csv"); train_windows = train_windows[train_windows["label"].isin(labels)].reset_index(drop=True)
    test_windows = generate_windows(test_samples, WindowConfig(2.0, 1.0)); test_windows = test_windows.reset_index(drop=True)
    summaries = []
    for config_path in sorted((ROOT / "configs/final").glob("*.yaml")):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")); experiment_id = config["experiment_id"]
        if config["model"]["name"] == "mlp": summaries.append(_evaluate_mlp(experiment_id, config, train_samples, train_windows, test_samples, test_windows, labels))
        else: summaries.append(_evaluate_classifier(experiment_id, config, train_samples, train_windows, test_samples, test_windows, labels))
    table = pd.DataFrame(summaries); output = ROOT / "results/tables"; output.mkdir(parents=True, exist_ok=True); table.to_csv(output / "final_holdout.csv", index=False); (ROOT / "results/tables/final_unseen_label_summary.json").write_text(json.dumps({"test_window_count": len(test_windows), "known_test_window_count": int(test_windows["label"].isin(labels).sum()), "unseen_test_window_count": int((~test_windows["label"].isin(labels)).sum()), "unseen_labels": sorted(set(test_windows.loc[~test_windows["label"].isin(labels), "label"]))}, indent=2) + "\n", encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()