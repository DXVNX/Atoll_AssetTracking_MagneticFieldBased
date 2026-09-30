import json

import pandas as pd
import yaml

from scripts.build_classical_results import collect_classical_runs


def test_collect_classical_runs_preserves_failed_metadata(tmp_path):
    config_dir = tmp_path / "configs/experiments/classical"
    run_dir = tmp_path / "results/runs/CLS_qda_body_xyz_00/development_s42_test"
    config_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    config = {
        "features": {"name": "body_xyz_magnitude", "schema_version": "body_xyz_magnitude@1"},
        "model": {"name": "qda", "parameters": {"reg_param": 0.1}},
        "preprocessing": {"scaler": "standard"},
    }
    (config_dir / "CLS_qda_body_xyz_00.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run_dir / "run_metadata.json").write_text(json.dumps({
        "status": "FAILED",
        "error": {"message": "ill defined covariance"},
    }), encoding="utf-8")

    result = collect_classical_runs(tmp_path)

    assert isinstance(result, pd.DataFrame)
    assert result.loc[0, "status"] == "FAILED"
    assert result.loc[0, "error"] == "ill defined covariance"