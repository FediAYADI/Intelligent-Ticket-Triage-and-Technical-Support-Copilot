"""Classification evaluation: per-head metrics + confusion matrices."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

from src.classification.model import LABEL_COLUMNS


def metrics_report(y_true: dict[str, np.ndarray],
                   y_pred: dict[str, np.ndarray],
                   label_maps: dict[str, dict[str, int]]) -> dict:
    """y_true/y_pred contain integer label indices; label_maps map label->idx."""
    inv = {c: {i: lab for lab, i in m.items()} for c, m in label_maps.items()}
    report = {}
    for col in LABEL_COLUMNS:
        yt = [inv[col][int(i)] for i in y_true[col]]
        yp = [inv[col][int(i)] for i in y_pred[col]]
        report[col] = {
            "accuracy": float(accuracy_score(yt, yp)),
            "precision_macro": float(precision_score(
                yt, yp, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(
                yt, yp, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(
                yt, yp, average="macro", zero_division=0)),
            "per_class": classification_report(
                yt, yp, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(
                yt, yp, labels=sorted(set(yt) | set(yp))).tolist(),
            "labels": sorted(set(yt) | set(yp)),
        }
    return report
