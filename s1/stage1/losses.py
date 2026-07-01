"""
Loss functions for imbalanced binary classification.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class focal loss.

    Useful when malicious label=1 is rare.
    """

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.alpha,
            reduction="none",
        )
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce

        # print("[INFO] losses.py ------ forward")

        return loss.mean()


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

# def compute_class_alpha(labels: List[int], num_classes: int = 2) -> torch.Tensor:
#     """
#     修正的类别权重计算 - 少数类获得更大权重
#     """
#     counts = np.bincount(np.array(labels, dtype=int), minlength=num_classes)
#     counts = np.maximum(counts, 1)
#
#     # 修改：少数类权重大，多数类权重小
#     total = counts.sum()
#     alpha = total / (num_classes * counts)  # 反比于频率
#
#     # 归一化
#     alpha = alpha / alpha.sum()
#
#     print(f"[INFO] Class distribution: {counts}")
#     print(f"[INFO] Computed alpha weights: {alpha}")
#
#     return torch.tensor(alpha, dtype=torch.float32)


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

class AsymmetricFocalLoss(nn.Module):
    """
    Binary asymmetric focal loss for 2-class softmax outputs.
    target: 0/1
    """

    def __init__(
            self,
            gamma_pos=0.0,
            gamma_neg=2.0,
            alpha_pos=0.55,
            eps=1e-8,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha_pos = alpha_pos
        self.eps = eps

    def forward(self, logits, target):
        prob = torch.softmax(logits, dim=-1)[:, 1]
        target = target.float()

        prob = torch.clamp(prob, self.eps, 1.0 - self.eps)

        pos_loss = -target * ((1.0 - prob) ** self.gamma_pos) * torch.log(prob)
        neg_loss = -(1.0 - target) * (prob ** self.gamma_neg) * torch.log(1.0 - prob)

        loss = self.alpha_pos * pos_loss + (1.0 - self.alpha_pos) * neg_loss
        return loss.mean()

class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss using effective number of samples.
    Suitable for imbalanced binary classification.
    """

    def __init__(
        self,
        labels,
        num_classes=2,
        beta=0.999,
        gamma=1.5,
        label_smoothing=0.0,
    ):
        super().__init__()

        counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes)
        counts = np.maximum(counts, 1)

        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num

        # normalize to mean 1, more stable than sum-to-1
        weights = weights / np.mean(weights)

        self.alpha = torch.tensor(weights, dtype=torch.float32)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

        print("[INFO] ClassBalancedFocalLoss counts:", counts)
        print("[INFO] ClassBalancedFocalLoss weights:", weights)

    def forward(self, logits, target):
        alpha = self.alpha.to(logits.device)

        ce = F.cross_entropy(
            logits,
            target,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        pt = torch.exp(-ce)
        at = alpha.gather(0, target)

        loss = at * ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()

class HardNegativeMiningCELoss(nn.Module):
    """
    Keep all positive samples and hardest negative samples in each batch.
    Useful for reducing false positives.
    """

    def __init__(self, neg_keep_ratio=0.30, label_smoothing=0.0):
        super().__init__()
        self.neg_keep_ratio = neg_keep_ratio
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits,
            target,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        pos_mask = target == 1
        neg_mask = target == 0

        pos_loss = ce[pos_mask]
        neg_loss = ce[neg_mask]

        losses = []

        if pos_loss.numel() > 0:
            losses.append(pos_loss)

        if neg_loss.numel() > 0:
            k = max(1, int(self.neg_keep_ratio * neg_loss.numel()))
            hard_neg_loss, _ = torch.topk(neg_loss, k=k, largest=True)
            losses.append(hard_neg_loss)

        if len(losses) == 0:
            return ce.mean()

        return torch.cat(losses).mean()