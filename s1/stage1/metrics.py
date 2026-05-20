# metrics.py
"""
Evaluation metrics — 支持多分类
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_score, loss=None, threshold=None, num_classes=None):
    """
    计算分类指标

    Args:
        y_true: 真实标签 (1D array)
        y_pred: 预测标签 (1D array)
        y_score: 预测概率 (1D or 2D array)
        loss: 损失值 (optional)
        threshold: 决策阈值 (optional)
        num_classes: 类别数 (optional)

    Returns:
        dict: 包含所有指标的字典
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    # 自动推断类别数
    if num_classes is None:
        num_classes = len(np.unique(np.concatenate([y_true, y_pred])))

    # 确保 y_score 是正确的二维数组
    y_score = np.array(y_score)
    if y_score.ndim == 1:
        # 对于二分类，构建两个类别的概率
        y_score_2d = np.zeros((len(y_score), max(2, num_classes)))
        y_score_2d[:, 1] = y_score
        y_score_2d[:, 0] = 1 - y_score
        y_score = y_score_2d

    # ============ 基础指标 ============
    acc = accuracy_score(y_true, y_pred)

    # 使用 f1_score 直接计算各种平均的 F1
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # 使用 precision_recall_fscore_support 获取精确度和召回率
    precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    precision_weighted, recall_weighted, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    # ============ 各类别指标 ============
    # per_class_f1 返回每个类别的 F1
    per_class_f1_array = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = {str(i): float(f1) for i, f1 in enumerate(per_class_f1_array)}

    # ============ 类别1 (恶意流量) 的指标 ============
    if num_classes == 2:
        # 二分类：使用 pos_label=1
        precision_1, recall_1, f1_1, _ = precision_recall_fscore_support(
            y_true, y_pred, pos_label=1, average='binary', zero_division=0
        )
        precision_label1 = float(precision_1)
        recall_label1 = float(recall_1)
        f1_label1 = float(f1_1)
    else:
        # 多分类：取类别1的指标
        precisions, recalls, f1s, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        if 1 < len(precisions):
            precision_label1 = float(precisions[1])
            recall_label1 = float(recalls[1])
            f1_label1 = float(f1s[1])
        else:
            # 如果标签中不包含类别1
            precision_label1 = 0.0
            recall_label1 = 0.0
            f1_label1 = 0.0

    # ============ AUC 计算 ============
    try:
        if num_classes == 2:
            # 二分类：使用类别1的概率 (y_score[:, 1])
            if y_score.shape[1] >= 2:
                auc = float(roc_auc_score(y_true, y_score[:, 1]))
            else:
                auc = float(roc_auc_score(y_true, y_score.ravel()))
        else:
            # 多分类：使用 OvR
            auc = float(roc_auc_score(
                y_true,
                y_score,
                multi_class='ovr',
                average='macro',
                labels=list(range(num_classes))
            ))
    except Exception as e:
        print(f"[WARNING] AUC calculation failed: {e}")
        # 打印调试信息
        print(f"  y_true shape: {y_true.shape}, unique: {np.unique(y_true)}")
        print(f"  y_score shape: {y_score.shape}")
        auc = 0.0

    # ============ 混淆矩阵 ============
    cm = confusion_matrix(y_true, y_pred)

    # ============ 构建结果字典 ============
    metrics = {
        'accuracy': float(acc),
        'macro_precision': float(precision_macro),
        'macro_recall': float(recall_macro),
        'macro_f1': float(f1_macro),
        'weighted_precision': float(precision_weighted),
        'weighted_recall': float(recall_weighted),
        'weighted_f1': float(f1_weighted),
        'precision_label1': precision_label1,
        'recall_label1': recall_label1,
        'f1_label1': f1_label1,
        'auc': auc,
        'per_class_f1': per_class_f1,
        'confusion_matrix': cm.tolist(),
        'num_classes': num_classes,
        'num_samples': len(y_true),
    }

    # 添加可选参数
    if loss is not None:
        metrics['loss'] = float(loss)
    if threshold is not None:
        metrics['threshold'] = float(threshold)

    return metrics