"""Build development windows and support artifacts from canonical UJI trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.loading import load_uji_trajectories
from src.data.splits import largest_complete_stratified_group_cv
from src.data.support import build_label_support
from src.data.windows import WindowConfig, generate_windows

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "external" / "ujiindoorloc_mag"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DATASET)
    parser.add_argument("--window-length", type=float, default=2.0)
    parser.add_argument("--stride", type=float, default=1.0)
    parser.add_argument("--final-test", action="store_true", help="explicitly permit test access for support accounting only")
    args = parser.parse_args()
    train = load_uji_trajectories(args.dataset_dir)
    windows = generate_windows(train, WindowConfig(args.window_length, args.stride))
    processed = args.dataset_dir / "processed"
    manifests = args.dataset_dir / "manifests"
    processed.mkdir(parents=True, exist_ok=True)
    token = f"{args.window_length:g}s_stride_{args.stride:g}s".replace(".", "p")
    windows.to_csv(processed / f"window_manifest_{token}.csv", index=False)
    empty_test = train.iloc[0:0].copy()
    support, scopes = build_label_support(train, empty_test, cv_folds=5, development_windows=windows)
    cv_windows = windows[windows["label"].isin(scopes["cv_eligible"])].reset_index(drop=True)
    fold_count, splits, feasibility = largest_complete_stratified_group_cv(cv_windows, maximum_splits=5)
    split_records = []
    for fold, (train_idx, validation_idx) in enumerate(splits):
        for partition, indices in (("train", train_idx), ("validation", validation_idx)):
            for group_id in sorted(cv_windows.iloc[indices]["group_id"].unique()):
                split_records.append({"fold": fold, "partition": partition, "group_id": group_id})
    import pandas as pd
    pd.DataFrame(split_records).to_csv(manifests / "development_group_folds.csv", index=False)
    summary = {
        "development_windows": len(windows),
        "cv_eligible_windows": len(cv_windows),
        "development_groups": windows["group_id"].nunique(),
        "cv_eligible_labels": len(scopes["cv_eligible"]),
        "rare_labels": len(scopes["rare_labels"]),
        "folds": fold_count,
        "fold_feasibility_missing_label_counts": {
            str(count): [len(missing) for missing in folds] for count, folds in feasibility.items()
        },
    }
    if args.final_test:
        test = load_uji_trajectories(args.dataset_dir, source_split="test_public", allow_final_test=True)
        support, scopes = build_label_support(train, test, cv_folds=5, development_windows=windows)
        support.to_csv(manifests / "label_support.csv", index=False)
        (manifests / "label_scopes.json").write_text(json.dumps(scopes, indent=2) + "\n", encoding="utf-8")
        summary.update({name: len(values) for name, values in scopes.items()})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

