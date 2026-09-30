"""Strict YAML experiment configuration loading and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

REQUIRED_SECTIONS = (
    "experiment_id", "name", "dataset", "target", "window", "preprocessing",
    "features", "evaluation", "model", "outputs", "seed",
)
MODEL_PARAMETERS = {
    "majority": set(),
    "random_prior": set(),
    "nearest_centroid": {"metric", "shrink_threshold"},
    "knn": {"n_neighbors", "weights", "metric", "p"},
    "logistic_regression": {"C", "class_weight", "solver", "multi_class"},
    "lda": {"solver", "shrinkage"},
    "qda": {"reg_param"},
    "linear_svm": {"C", "class_weight"},
    "rbf_svm": {"C", "gamma", "class_weight"},
    "poly_svm": {"C", "degree", "gamma", "coef0", "class_weight"},
    "decision_tree": {"max_depth", "min_samples_leaf", "class_weight"},
    "random_forest": {"n_estimators", "max_depth", "min_samples_leaf", "max_features", "class_weight"},
    "extra_trees": {"n_estimators", "max_depth", "min_samples_leaf", "max_features", "class_weight"},
    "hist_gradient_boosting": {"learning_rate", "max_leaf_nodes", "l2_regularization", "max_iter"},
}


class ConfigError(ValueError):
    pass


def validate_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ConfigError("configuration must be a mapping")
    missing = [key for key in REQUIRED_SECTIONS if key not in config]
    if missing:
        raise ConfigError(f"missing required config sections: {missing}")
    allowed_top = set(REQUIRED_SECTIONS) | {"notes"}
    if unknown := sorted(set(config) - allowed_top):
        raise ConfigError(f"unknown top-level config fields: {unknown}")
    for section in ("dataset", "target", "window", "preprocessing", "features", "evaluation", "model", "outputs"):
        if not isinstance(config[section], dict):
            raise ConfigError(f"{section} must be a mapping")
    model_name = config["model"].get("name")
    if model_name not in MODEL_PARAMETERS:
        raise ConfigError(f"unsupported model: {model_name!r}")
    parameters = config["model"].get("parameters", {})
    if not isinstance(parameters, dict):
        raise ConfigError("model.parameters must be a mapping")
    if unknown := sorted(set(parameters) - MODEL_PARAMETERS[model_name]):
        raise ConfigError(f"unknown parameters for {model_name}: {unknown}")
    if model_name == "knn":
        if int(parameters.get("n_neighbors", 0)) <= 0:
            raise ConfigError("knn n_neighbors must be positive")
        if parameters.get("weights") not in {"uniform", "distance"}:
            raise ConfigError("knn weights must be uniform or distance")
        if parameters.get("metric") not in {"euclidean", "manhattan", "minkowski"}:
            raise ConfigError("unsupported knn metric")
        if parameters.get("metric") == "minkowski" and float(parameters.get("p", 0)) != 3:
            raise ConfigError("this experiment family requires minkowski p=3")
        if parameters.get("metric") != "minkowski" and "p" in parameters:
            raise ConfigError("knn p is only valid with metric=minkowski")
    required_nested = {
        "dataset": {"id", "path"}, "target": {"column"},
        "window": {"length_s", "stride_s"}, "features": {"name", "schema_version"},
        "evaluation": {"split_manifest", "group_column", "strategy"},
        "outputs": {"root"},
    }
    for section, fields in required_nested.items():
        if missing_nested := sorted(fields - set(config[section])):
            raise ConfigError(f"{section} missing fields: {missing_nested}")
    if config["target"]["column"] != "label":
        raise ConfigError("baseline harness currently supports target.column=label only")
    if float(config["window"]["length_s"]) <= 0 or float(config["window"]["stride_s"]) <= 0:
        raise ConfigError("window length and stride must be positive")
    int(config["seed"])
    return config


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return validate_config(yaml.safe_load(handle))


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

