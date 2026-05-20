"""
Evaluation metrics.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def classification_metrics(y_true: List[int], y_pred: List[int], y_score: List[float]) -> Dict:
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    y_score = np.array(y_score, dtype=float)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_score)
    else:
        auc = float("nan")

    print("[INFO] metircs.py ------ classification_metrics")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_label1": float(precision),
        "recall_label1": float(recall),
        "f1_label1": float(f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "auc": float(auc),
        "confusion_matrix_0_1": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "num_samples": int(len(y_true)),
        "num_label_1": int((y_true == 1).sum()),
    }
