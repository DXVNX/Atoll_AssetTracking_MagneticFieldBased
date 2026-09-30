import copy

import pytest

from src.experiments.config import ConfigError, validate_config


def valid_config():
    return {
        "experiment_id": "E_TEST", "name": "test",
        "dataset": {"id": "ujiindoorloc_mag", "path": "data"},
        "target": {"column": "label"}, "window": {"length_s": 2.0, "stride_s": 1.0},
        "preprocessing": {"scaler": "none"},
        "features": {"name": "magnitude_stats", "schema_version": "magnitude_stats@1"},
        "evaluation": {"strategy": "saved", "split_manifest": "folds.csv", "group_column": "trajectory_id"},
        "model": {"name": "majority", "parameters": {}},
        "outputs": {"root": "results"}, "seed": 42,
    }


def test_config_requires_every_section():
    config = valid_config(); del config["features"]
    with pytest.raises(ConfigError, match="missing required"):
        validate_config(config)


def test_unknown_model_parameter_fails():
    config = valid_config(); config["model"]["parameters"] = {"n_estimator": 10}
    with pytest.raises(ConfigError, match="unknown parameters"):
        validate_config(config)


def test_unknown_top_level_field_fails():
    config = copy.deepcopy(valid_config()); config["typo"] = True
    with pytest.raises(ConfigError, match="unknown top-level"):
        validate_config(config)


def test_knn_parameters_are_strict_and_validated():
    config = valid_config()
    config["model"] = {"name": "knn", "parameters": {"n_neighbors": 5, "weights": "distance", "metric": "minkowski", "p": 3}}
    assert validate_config(config)["model"]["parameters"]["p"] == 3
    config["model"]["parameters"]["leaf_size"] = 99
    with pytest.raises(ConfigError, match="unknown parameters"):
        validate_config(config)


def test_knn_p_only_applies_to_minkowski():
    config = valid_config()
    config["model"] = {"name": "knn", "parameters": {"n_neighbors": 3, "weights": "uniform", "metric": "euclidean", "p": 3}}
    with pytest.raises(ConfigError, match="only valid"):
        validate_config(config)

