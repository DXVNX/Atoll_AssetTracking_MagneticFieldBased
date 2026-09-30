"""Generate final-holdout figures and a compact dataset summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output = ROOT / "results"
    final = pd.read_csv(output / "tables/final_holdout.csv")
    figure_dir = output / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5)); axis.bar(final["model"], final["macro_f1"]); axis.set_ylabel("Closed-set test macro-F1"); axis.set_title("Frozen candidates: final holdout comparison"); figure.tight_layout(); figure.savefig(figure_dir / "final_holdout_comparison.png", dpi=200); plt.close(figure)
    for row in final.itertuples():
        matrix_path = output / "runs" / row.experiment_id / "final_s42/confusion_matrix.csv"
        if not matrix_path.exists():
            continue
        matrix = pd.read_csv(matrix_path, index_col=0)
        figure, axis = plt.subplots(figsize=(8, 7)); image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap="Blues"); figure.colorbar(image, ax=axis); axis.set_title(f"{row.model}: closed-set final confusion matrix"); axis.set_xlabel("Predicted label"); axis.set_ylabel("True label"); figure.tight_layout(); figure.savefig(figure_dir / f"final_confusion_{row.experiment_id}.png", dpi=180); plt.close(figure)
    train_manifest = pd.read_csv(ROOT / "data/external/ujiindoorloc_mag/manifests/trajectory_manifest.csv")
    windows = pd.read_csv(ROOT / "data/external/ujiindoorloc_mag/processed/window_manifest_2s_stride_1s.csv")
    summary = pd.DataFrame([
        {"scope": "train_public", "trajectories": int((train_manifest["source_split"] == "train_public").sum()), "windows": len(windows), "sensor_rate_hz_median": 10.0, "sensors": "magnetometer XYZ; linear accelerometer XYZ; orientation"},
        {"scope": "test_public", "trajectories": int((train_manifest["source_split"] == "test_public").sum()), "windows": int(json.loads((output / "tables/final_unseen_label_summary.json").read_text(encoding="utf-8"))["test_window_count"]), "sensor_rate_hz_median": 10.0, "sensors": "magnetometer XYZ; linear accelerometer XYZ; orientation"},
    ])
    summary.to_csv(output / "tables/dataset_summary.csv", index=False)
    print("generated final holdout figures and dataset summary")


if __name__ == "__main__":
    main()