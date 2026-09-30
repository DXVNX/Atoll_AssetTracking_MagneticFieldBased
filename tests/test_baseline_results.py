import json

from scripts.build_baseline_results import collect_baselines


def test_collect_baselines_reads_grouped_run_metrics(tmp_path):
    run_dir = tmp_path / "results/runs/BASE_majority_body_xyz/development_s42_test"
    run_dir.mkdir(parents=True)
    (run_dir / "config_resolved.yaml").write_text(
        "features:\n  name: body_xyz_magnitude\n", encoding="utf-8"
    )
    (run_dir / "run_metadata.json").write_text(json.dumps({
        "experiment_id": "BASE_majority_body_xyz",
        "model": "majority",
        "status": "COMPLETED",
        "metrics_summary": {
            "accuracy_mean": 0.1,
            "balanced_accuracy_mean": 0.03,
            "macro_f1_mean": 0.01,
        },
    }), encoding="utf-8")

    result = collect_baselines(tmp_path)

    assert result.loc[0, "feature_set"] == "body_xyz_magnitude"
    assert result.loc[0, "macro_f1"] == 0.01