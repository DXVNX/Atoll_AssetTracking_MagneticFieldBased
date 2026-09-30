"""Minimal baseline estimators used to validate the harness."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_prior_ = counts / counts.sum()
        self.majority_ = self.classes_[np.argmax(counts)]
        return self

    def predict(self, X):
        return np.repeat(self.majority_, len(X))

    def predict_proba(self, X):
        return np.tile(self.class_prior_, (len(X), 1))


class RandomPriorClassifier(MajorityClassifier):
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def predict(self, X):
        return np.random.default_rng(self.random_state).choice(self.classes_, size=len(X), p=self.class_prior_)


def build_baseline(name: str, parameters: dict, seed: int):
    if name == "majority":
        return MajorityClassifier()
    if name == "random_prior":
        return RandomPriorClassifier(random_state=seed)
    if name == "nearest_centroid":
        return NearestCentroid(**parameters)
    if name == "knn":
        return KNeighborsClassifier(**parameters)
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=seed, **parameters)
    if name == "lda":
        return LinearDiscriminantAnalysis(**parameters)
    if name == "qda":
        return QuadraticDiscriminantAnalysis(**parameters)
    if name == "linear_svm":
        return SVC(kernel="linear", probability=False, random_state=seed, **parameters)
    if name == "rbf_svm":
        return SVC(kernel="rbf", probability=False, random_state=seed, **parameters)
    if name == "poly_svm":
        return SVC(kernel="poly", probability=False, random_state=seed, **parameters)
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed, **parameters)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=1, **parameters)
    if name == "extra_trees":
        return ExtraTreesClassifier(random_state=seed, n_jobs=1, **parameters)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed, **parameters)
    raise ValueError(f"unsupported baseline: {name}")

