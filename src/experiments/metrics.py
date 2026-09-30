"""Standard classification metrics for window and trajectory-balanced reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(y_true, y_pred, *, labels=None, scores=None, score_labels=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(labels if labels is not None else sorted(set(y_true) | set(y_pred)))
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    per = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]), "macro_recall": float(macro[1]), "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
        "per_class": {str(label): {"precision": float(per[0][i]), "recall": float(per[1][i]), "f1": float(per[2][i]), "support": int(per[3][i])} for i, label in enumerate(labels)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "confusion_matrix_labels": [str(label) for label in labels],
    }
    if scores is not None and score_labels is not None and len(score_labels) >= 2:
        score_labels = np.asarray(score_labels)
        top_indices = np.argsort(np.asarray(scores), axis=1)[:, -2:]
        top_labels = score_labels[top_indices]
        result["top2_accuracy"] = float(np.mean([truth in choices for truth, choices in zip(y_true, top_labels)]))
    return result


def trajectory_balanced_metrics(predictions: pd.DataFrame) -> dict:
    accuracies, macro_f1s = [], []
    labels = sorted(set(predictions["true_label"]) | set(predictions["predicted_label"]))
    for _, rows in predictions.groupby("trajectory_id"):
        accuracies.append(accuracy_score(rows["true_label"], rows["predicted_label"]))
        macro_f1s.append(precision_recall_fscore_support(rows["true_label"], rows["predicted_label"], labels=labels, average="macro", zero_division=0)[2])
    return {
        "trajectory_accuracy_mean": float(np.mean(accuracies)),
        "trajectory_macro_f1_mean": float(np.mean(macro_f1s)),
        "trajectory_count": len(accuracies),
    }

