"""Classification metrics for binary / multiclass tasks.

All functions accept numpy arrays:
  y_true : (N,) integer class labels
  y_prob : (N, C) class probabilities (softmax) OR (N,) positive-class prob
"""
from typing import Dict

import numpy as np


def _as_prob_matrix(y_prob: np.ndarray, num_classes: int) -> np.ndarray:
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 1:
        # positive-class probability for binary
        y_prob = np.stack([1.0 - y_prob, y_prob], axis=1)
    return y_prob


def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    """Top-label ECE."""
    y_prob = np.asarray(y_prob)
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            ece += (m.sum() / n) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def compute_metrics(y_true, y_prob, num_classes: int = 2) -> Dict[str, float]:
    """Return a dict of AUC / accuracy / F1 / precision / recall / ECE."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    P = _as_prob_matrix(y_prob, num_classes)
    y_pred = P.argmax(axis=1)

    out: Dict[str, float] = {}
    # AUC
    try:
        if num_classes == 2:
            out["auc"] = float(roc_auc_score(y_true, P[:, 1]))
        else:
            out["auc"] = float(
                roc_auc_score(y_true, P, multi_class="ovr", average="macro")
            )
    except ValueError:
        out["auc"] = float("nan")  # single-class batch

    avg = "binary" if num_classes == 2 else "macro"
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["f1"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
    out["precision"] = float(
        precision_score(y_true, y_pred, average=avg, zero_division=0)
    )
    out["recall"] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
    out["ece"] = expected_calibration_error(y_true, P)
    return out


def confusion(y_true, y_prob, num_classes: int = 2):
    from sklearn.metrics import confusion_matrix

    P = _as_prob_matrix(y_prob, num_classes)
    y_pred = P.argmax(axis=1)
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def per_class_metrics(y_true, y_prob, num_classes: int = 2) -> Dict[str, Dict]:
    from sklearn.metrics import precision_recall_fscore_support

    P = _as_prob_matrix(y_prob, num_classes)
    y_pred = P.argmax(axis=1)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    return {
        str(c): {
            "precision": float(prec[c]),
            "recall": float(rec[c]),
            "f1": float(f1[c]),
            "support": int(sup[c]),
        }
        for c in range(num_classes)
    }
