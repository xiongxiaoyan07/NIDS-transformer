from __future__ import annotations

from typing import Iterable, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_alpha(labels: List[int], num_classes: int = 2) -> torch.Tensor:
    """
    计算类别权重 - 少数类获得更大权重

    对于极度不平衡数据，使用更激进的权重策略
    """
    counts = np.bincount(np.array(labels, dtype=int), minlength=num_classes)
    counts = np.maximum(counts, 1)

    # 计算不平衡比率
    majority_count = counts.max()
    minority_count = counts.min()
    ratio = majority_count / minority_count

    print(f"[INFO] Class distribution: {counts}")
    print(f"[INFO] Imbalance ratio: {ratio:.1f}:1")

    # 根据不平衡程度选择权重策略
    if ratio > 20:
        # 极度不平衡：使用更激进的权重
        alpha = np.array([1.0 / counts[0], 1.0 / counts[1]])
        alpha = alpha / alpha.sum()
        print(f"[INFO] 极度不平衡，使用反比权重")
    elif ratio > 5:
        # 中度不平衡
        total = counts.sum()
        alpha = total / (num_classes * counts)
        print(f"[INFO] 中度不平衡，使用标准平衡权重")
    else:
        # 接近平衡
        total = counts.sum()
        alpha = total / (num_classes * counts)
        print(f"[INFO] 接近平衡，使用轻微加权权重")

    # 归一化
    alpha = alpha / alpha.sum()

    print(f"[INFO] Computed alpha weights: {alpha}")
    print(f"[INFO] Few class weight / Many class weight: {alpha[1] / alpha[0]:.2f}")

    return torch.tensor(alpha, dtype=torch.float32)

class FocalLossWithLabelSmoothing(nn.Module):
    """
    结合Focal Loss和标签平滑的损失函数
    增强数值稳定性
    """

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.eps = eps

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) logits
            targets: (batch_size,) class indices
        """
        num_classes = inputs.size(-1)

        # 1. 计算log概率和概率（增加数值稳定性）
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        # 裁剪概率避免log(0)
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)

        # 2. 获取真实类别的概率
        probs_gt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # 3. 计算focal weight
        focal_weight = (1 - probs_gt) ** self.gamma

        # 4. 应用类别权重alpha
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_weight = self.alpha[targets]
            focal_weight = focal_weight * alpha_weight

        # 5. 计算标签平滑的交叉熵
        # 使用PyTorch的内置函数，更稳定
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction='none',
            label_smoothing=self.label_smoothing
        )

        # 6. 应用focal weight并求平均
        loss = (focal_weight * ce_loss).mean()

        # 7. 检查是否有NaN
        if torch.isnan(loss):
            print(f"[WARNING] NaN detected in loss!")
            print(f"  probs_gt range: [{probs_gt.min():.4f}, {probs_gt.max():.4f}]")
            print(f"  focal_weight range: [{focal_weight.min():.4f}, {focal_weight.max():.4f}]")
            print(f"  ce_loss range: [{ce_loss.min():.4f}, {ce_loss.max():.4f}]")
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        return loss