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


def set_seed(seed: int = 42, deterministic=True):
    """
    Set random seeds for full reproducibility.

    关键点：
    1. Python random + numpy + PyTorch 全覆盖
    2. cudnn deterministic 模式（略微降低性能但保证可复现）
    3. 环境变量设置
    """

    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU

    # CuDNN
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 注意：某些 CUDA 操作在 deterministic 模式下可能变慢
        # 但这是可复现性的代价
    else:
        torch.backends.cudnn.benchmark = True

    # 环境变量
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 允许 CuBLAS 确定性

    # PyTorch 2.0+ 的确定性算法（需要 CUDA 10.2+）
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True)

    print(f"[INFO] 随机种子已设置: {seed}")


class SeedContext:
    """
    Context manager for temporary seed changes

    用法：
    with SeedContext(42):
        # 这段代码的随机性由 seed 42 控制
        train_model()
    # 离开 context 后恢复原来的随机状态
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        set_seed(self.seed)
        return self

    def __exit__(self, *args):
        if self.state:
            random.setstate(self.state['python'])
            np.random.set_state(self.state['numpy'])
            torch.set_rng_state(self.state['torch'])
            if self.state['cuda']:
                torch.cuda.set_rng_state_all(self.state['cuda'])

def worker_init_fn(worker_id: int):
    """
    DataLoader worker 的初始化函数
    确保每个 worker 的随机状态是可确定的
    """
    # 基于 worker_id 和基础种子生成 worker 特定的种子
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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