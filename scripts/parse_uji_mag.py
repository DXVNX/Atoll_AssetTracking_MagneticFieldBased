"""Parse frozen UJIIndoorLoc-Mag raw trajectories into canonical interim CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from src.adapters.uji_mag import OUTPUT_COLUMNS, UJIParseError, canonical_endpoint, parse_uji_trajectory

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "external" / "ujiindoorloc_mag"


def run(dataset_dir: Path = DATASET, limit: int | None = None, verbose: bool = False) -> dict[str, int]:
    raw = dataset_dir / "raw"
    manifests = dataset_dir / "manifests"
    interim = dataset_dir / "interim"
    rows = list(csv.DictReader((manifests / "raw_manifest.csv").open(encoding="utf-8", newline="")))
    if limit is not None:
        rows = rows[:limit]
    interim.mkdir(parents=True, exist_ok=True)
    trajectory_rows, segment_observations, failures = [], [], []
    for row in rows:
        path = raw / Path(row["relative_path"])
        try:
            trajectory = parse_uji_trajectory(path, row["trajectory_id"], row["source_split"], row["trajectory_type"])
        except UJIParseError as exc:
            failures.append({"trajectory_id": row["trajectory_id"], "relative_path": row["relative_path"], "line_number": exc.line_number or "", "reason": exc.reason})
            continue
        output = interim / row["source_split"] / f"{row['trajectory_id']}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        trajectory.samples.to_csv(output, index=False)
        duration = float(trajectory.samples["elapsed_s"].iloc[-1])
        trajectory_rows.append({
            "trajectory_id": trajectory.trajectory_id, "source_split": trajectory.source_split,
            "trajectory_type": trajectory.trajectory_type, "n_samples": len(trajectory.samples),
            "duration_s": f"{duration:.3f}", "n_segments": len(trajectory.segments),
            "sampling_rate_est_hz": f"{trajectory.sampling_rate_est_hz:.6f}" if trajectory.sampling_rate_est_hz else "",
            "parse_warnings": ";".join(trajectory.warnings),
        })
        for segment in trajectory.segments:
            a, b = sorted((canonical_endpoint(segment.start_lat, segment.start_lon), canonical_endpoint(segment.end_lat, segment.end_lon)))
            segment_observations.append({"segment_id": segment.segment_id, "a": a, "b": b, "split": trajectory.source_split, "trajectory_id": trajectory.trajectory_id, "samples": segment.last_sample - segment.first_sample + 1})
        if verbose:
            print(f"parsed {trajectory.trajectory_id}: {len(trajectory.samples)} samples, {len(trajectory.segments)} segments")

    def write_csv(path: Path, fieldnames: list[str], records: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(records)

    write_csv(manifests / "trajectory_manifest.csv", ["trajectory_id", "source_split", "trajectory_type", "n_samples", "duration_s", "n_segments", "sampling_rate_est_hz", "parse_warnings"], trajectory_rows)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for observation in segment_observations:
        grouped[observation["segment_id"]].append(observation)
    labels = []
    for segment_id, observations in sorted(grouped.items()):
        first = observations[0]
        labels.append({
            "segment_id": segment_id, "endpoint_a_lat": first["a"][0], "endpoint_a_lon": first["a"][1],
            "endpoint_b_lat": first["b"][0], "endpoint_b_lon": first["b"][1],
            "train_trajectory_count": len({o["trajectory_id"] for o in observations if o["split"] == "train_public"}),
            "test_trajectory_count": len({o["trajectory_id"] for o in observations if o["split"] == "test_public"}),
            "sample_count": sum(o["samples"] for o in observations),
        })
    write_csv(manifests / "label_manifest.csv", ["segment_id", "endpoint_a_lat", "endpoint_a_lon", "endpoint_b_lat", "endpoint_b_lon", "train_trajectory_count", "test_trajectory_count", "sample_count"], labels)
    write_csv(manifests / "parse_failures.csv", ["trajectory_id", "relative_path", "line_number", "reason"], failures)
    summary = {"attempted": len(rows), "parsed": len(trajectory_rows), "failed": len(failures), "labels": len(labels)}
    (manifests / "parse_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DATASET)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    summary = run(args.dataset_dir, args.limit, args.verbose)
    print(json.dumps(summary, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
