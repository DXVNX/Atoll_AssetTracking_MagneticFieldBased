import numpy as np
import pytest

from src.experiments.baselines import build_baseline


@pytest.mark.parametrize("name,parameters", [
    ("logistic_regression", {"C": 1.0, "class_weight": None}),
    ("lda", {"solver": "svd"}),
    ("qda", {"reg_param": 0.1}),
    ("linear_svm", {"C": 1.0}),
    ("rbf_svm", {"C": 1.0, "gamma": "scale"}),
    ("poly_svm", {"C": 1.0, "degree": 2, "gamma": "scale", "coef0": 1}),
    ("decision_tree", {"max_depth": 3, "min_samples_leaf": 1}),
])
def test_classical_factory_fits(name, parameters):
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])
    assert len(build_baseline(name, parameters, 42).fit(x, y).predict(x)) == len(y)
