from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def get_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)

def binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if len(np.unique(y_true)) > 1:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["auroc"] = None

        try:
            metrics["auprc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["auprc"] = None
    else:
        metrics["auroc"] = None
        metrics["auprc"] = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = cm.astype(int).tolist()
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])
    return metrics

def metric_value(metrics_by_split: Dict[str, Dict[str, Any]], metric_name: str) -> float:
    if metric_name.startswith("val_"):
        key = metric_name[len("val_"):]
        value = metrics_by_split["val"].get(key)
    else:
        value = metrics_by_split["val"].get(metric_name)

    if value is None:
        return -float("inf")

    # We maximize score. For loss, use negative loss.
    if metric_name in {"val_loss", "loss"}:
        return -float(value)

    return float(value)