"""Helpers that make fold-local fitting explicit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone


@dataclass
class FoldTransformResult:
    transformer: object
    train: np.ndarray
    validation: np.ndarray


def fit_transform_fold(transformer: object, values: object, train_indices: object, validation_indices: object) -> FoldTransformResult:
    """Clone and fit a learned transform on training rows only."""
    fitted = clone(transformer)
    train_values = np.asarray(values)[np.asarray(train_indices)]
    validation_values = np.asarray(values)[np.asarray(validation_indices)]
    transformed_train = fitted.fit_transform(train_values)
    transformed_validation = fitted.transform(validation_values)
    return FoldTransformResult(fitted, transformed_train, transformed_validation)

