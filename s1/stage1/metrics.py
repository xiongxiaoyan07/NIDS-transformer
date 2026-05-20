"""
Evaluation metrics — 支持多分类
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    f1_score,
    classification_report,
)


def classification_metrics(y_true: List[int], y_pred: List[int], y_score: List[float]) -> Dict:
    """
    计算分类指标，支持二分类和多分类。

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_score: 预测概率（二分类时为类别1的概率，多分类时为所有类别的概率）
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    num_classes = len(np.unique(y_true))

    # ------------------------------------------------------------
    # 多分类指标
    # ------------------------------------------------------------
    # Macro 平均
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Weighted 平均
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # 各类别 F1
    per_class_f1 = f1_score(y_true, y_pred, average=None)

    # ------------------------------------------------------------
    # AUC 计算
    # ------------------------------------------------------------
    if num_classes == 2:
        # 二分类
        precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        # y_score 应该是类别1的概率
        y_score_arr = np.array(y_score)
        if y_score_arr.ndim == 2 and y_score_arr.shape[1] == 2:
            # 如果是二维概率，取类别1
            y_score_for_auc = y_score_arr[:, 1]
        else:
            y_score_for_auc = y_score_arr

        try:
            auc = float(roc_auc_score(y_true, y_score_for_auc))
        except:
            auc = float("nan")
    else:
        # 多分类：使用 One-vs-Rest
        precision_binary = float("nan")
        recall_binary = float("nan")
        f1_binary = float("nan")

        y_score_arr = np.array(y_score)
        if y_score_arr.ndim == 1:
            auc = float("nan")
        else:
            try:
                auc = float(roc_auc_score(
                    y_true, y_score_arr,
                    multi_class="ovr",
                    average="macro"
                ))
            except:
                auc = float("nan")

    # ------------------------------------------------------------
    # 混淆矩阵
    # ------------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)

    print("[INFO] metrics.py ------ classification_metrics (multi-class)")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # 多分类指标
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        # 二分类指标（仅在二分类时有意义）
        "precision_label1": float(precision_binary),
        "recall_label1": float(recall_binary),
        "f1_label1": float(f1_binary),
        # AUC
        "auc": float(auc),
        # 各类别 F1
        "per_class_f1": {str(i): float(f) for i, f in enumerate(per_class_f1)},
        # 混淆矩阵
        "confusion_matrix": cm.tolist(),
        # 统计信息
        "num_classes": num_classes,
        "num_samples": int(len(y_true)),
    }