#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flow-only baseline diagnosis for Stage1 scheme C.

Purpose:
    Test whether flow-level statistical features alone can reach:
        class1 precision >= 0.85
        class1 recall    >= 0.85
        class1 f1        >= 0.85

Models:
    1. Flow-only MLP
    2. XGBoost
    3. LightGBM

Input:
    Precomputed scheme-C npz files containing:
        flow_feats
        labels

Example:
    python -m stage1.flow_only_baselines \
      --train_npz /content/drive/MyDrive/s1/0701_C/0701_asymmetric_gated_fusion/precomputed/seqLen64headtrainC.npz \
      --val_npz   /content/drive/MyDrive/s1/0701_C/0701_asymmetric_gated_fusion/precomputed/seqLen64headvalC.npz \
      --test_npz  /content/drive/MyDrive/s1/0701_C/0701_asymmetric_gated_fusion/precomputed/seqLen64headtestC.npz \
      --out_dir   /content/drive/MyDrive/s1/0701_C/flow_only_diagnosis \
      --models mlp xgb lgbm \
      --loss cb_focal
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import log_loss

from .metrics import classification_metrics
from .losses import (
    FocalLossWithLabelSmoothing,
    ClassBalancedFocalLoss,
    AsymmetricFocalLoss,
    HardNegativeMiningCELoss,
)


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int = 130) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    print(f"[INFO] seed={seed}")


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)


def sanitize_features(x: np.ndarray, clip_value: float = 50.0) -> np.ndarray:
    """
    防止 flow_feats 中存在 NaN/Inf 或极端异常值。
    你的 flow_feats 已经经过 robust scaler + clipping，
    这里是额外保险。
    """
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    x = np.clip(x, -clip_value, clip_value)
    return x.astype(np.float32)


def load_flow_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)

    if "flow_feats" not in data:
        raise KeyError(
            f"{npz_path} does not contain 'flow_feats'. "
            f"Please use scheme C precomputed npz."
        )

    if "labels" not in data:
        raise KeyError(f"{npz_path} does not contain 'labels'.")

    x = sanitize_features(data["flow_feats"])
    y = np.asarray(data["labels"], dtype=np.int64)

    if len(x) != len(y):
        raise ValueError(
            f"Length mismatch in {npz_path}: "
            f"flow_feats={len(x)}, labels={len(y)}"
        )

    return x, y


def print_dataset_report(name: str, x: np.ndarray, y: np.ndarray) -> None:
    counts = np.bincount(y.astype(int), minlength=2)
    pos_ratio = counts[1] / max(counts.sum(), 1)

    print(f"\n[DATA] {name}")
    print(f"  X shape      : {x.shape}")
    print(f"  y shape      : {y.shape}")
    print(f"  class_counts : {{0: {counts[0]}, 1: {counts[1]}}}")
    print(f"  label1_ratio : {pos_ratio:.6f}")


def safe_binary_log_loss(y_true: np.ndarray, prob1: np.ndarray) -> float:
    prob1 = np.asarray(prob1, dtype=np.float64)
    prob1 = np.clip(prob1, 1e-7, 1.0 - 1e-7)
    y_score = np.stack([1.0 - prob1, prob1], axis=1)
    return float(log_loss(y_true, y_score, labels=[0, 1]))


# =============================================================================
# Metrics and threshold search
# =============================================================================

def metrics_at_threshold(
    y_true: np.ndarray,
    prob1: np.ndarray,
    threshold: float,
    loss: Optional[float] = None,
) -> Dict[str, Any]:
    prob1 = np.asarray(prob1, dtype=np.float64)
    y_pred = (prob1 >= threshold).astype(np.int64)
    y_score = np.stack([1.0 - prob1, prob1], axis=1)

    return classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        loss=loss,
        threshold=threshold,
        num_classes=2,
    )


def threshold_search(
    y_true: np.ndarray,
    prob1: np.ndarray,
    target_precision: float = 0.85,
    target_recall: float = 0.85,
    threshold_min: float = 0.001,
    threshold_max: float = 0.999,
    threshold_steps: int = 999,
) -> Dict[str, Any]:
    """
    在验证集上搜索阈值。

    输出三类结果：
    1. best_f1: class1 F1 最高的阈值
    2. best_min_pr: min(precision, recall) 最高的阈值
    3. best_feasible_085: 同时满足 P>=0.85 且 R>=0.85 的阈值，若不存在则为 None
    """
    thresholds = np.linspace(threshold_min, threshold_max, threshold_steps)

    loss = safe_binary_log_loss(y_true, prob1)

    rows: List[Dict[str, Any]] = []

    for th in thresholds:
        m = metrics_at_threshold(y_true, prob1, float(th), loss=loss)
        row = {
            "threshold": float(th),
            "precision_label1": float(m["precision_label1"]),
            "recall_label1": float(m["recall_label1"]),
            "f1_label1": float(m["f1_label1"]),
            "macro_f1": float(m["macro_f1"]),
            "pr_auc": float(m["pr_auc"]),
            "auc": float(m["auc"]),
            "min_pr": float(min(m["precision_label1"], m["recall_label1"])),
            "metrics": m,
        }
        rows.append(row)

    best_f1 = max(rows, key=lambda r: r["f1_label1"])
    best_min_pr = max(rows, key=lambda r: r["min_pr"])

    feasible = [
        r for r in rows
        if r["precision_label1"] >= target_precision
        and r["recall_label1"] >= target_recall
    ]

    best_feasible = None
    if len(feasible) > 0:
        best_feasible = max(feasible, key=lambda r: r["f1_label1"])

    return {
        "best_f1": best_f1,
        "best_min_pr": best_min_pr,
        "best_feasible_085": best_feasible,
        "num_feasible_085": len(feasible),
        "target_precision": target_precision,
        "target_recall": target_recall,
    }


def evaluate_prob_model(
    model_name: str,
    y_val: np.ndarray,
    val_prob1: np.ndarray,
    y_test: np.ndarray,
    test_prob1: np.ndarray,
    out_dir: str,
) -> Dict[str, Any]:
    """
    合法评估：
        只在 val 上选阈值，然后迁移到 test。

    诊断评估：
        test_oracle 只用于判断上限，不能作为正式结果写论文。
    """
    print(f"\n{'=' * 80}")
    print(f"[EVAL] {model_name}")
    print(f"{'=' * 80}")

    val_search = threshold_search(y_val, val_prob1)

    val_best = val_search["best_f1"]
    val_best_th = float(val_best["threshold"])

    test_loss = safe_binary_log_loss(y_test, test_prob1)

    test_at_val_best = metrics_at_threshold(
        y_true=y_test,
        prob1=test_prob1,
        threshold=val_best_th,
        loss=test_loss,
    )

    test_search = threshold_search(y_test, test_prob1)
    test_oracle = test_search["best_f1"]

    print("\n[VAL best class1-F1 threshold]")
    print(f"  threshold : {val_best_th:.6f}")
    print(f"  P1        : {val_best['precision_label1']:.4f}")
    print(f"  R1        : {val_best['recall_label1']:.4f}")
    print(f"  F1_1      : {val_best['f1_label1']:.4f}")
    print(f"  PR_AUC    : {val_best['pr_auc']:.4f}")

    print("\n[TEST at VAL threshold]")
    print(f"  threshold : {test_at_val_best['threshold']:.6f}")
    print(f"  P1        : {test_at_val_best['precision_label1']:.4f}")
    print(f"  R1        : {test_at_val_best['recall_label1']:.4f}")
    print(f"  F1_1      : {test_at_val_best['f1_label1']:.4f}")
    print(f"  Macro-F1  : {test_at_val_best['macro_f1']:.4f}")
    print(f"  PR_AUC    : {test_at_val_best['pr_auc']:.4f}")
    print(f"  AUC       : {test_at_val_best['auc']:.4f}")
    print(f"  CM        : {test_at_val_best['confusion_matrix']}")

    print("\n[TEST ORACLE - diagnostic only]")
    print(f"  threshold : {test_oracle['threshold']:.6f}")
    print(f"  P1        : {test_oracle['precision_label1']:.4f}")
    print(f"  R1        : {test_oracle['recall_label1']:.4f}")
    print(f"  F1_1      : {test_oracle['f1_label1']:.4f}")

    if val_search["best_feasible_085"] is not None:
        feasible_th = float(val_search["best_feasible_085"]["threshold"])
        test_at_feasible = metrics_at_threshold(
            y_true=y_test,
            prob1=test_prob1,
            threshold=feasible_th,
            loss=test_loss,
        )

        print("\n[VAL has threshold satisfying P>=0.85 and R>=0.85]")
        print(f"  threshold : {feasible_th:.6f}")
        print("[TEST at this feasible threshold]")
        print(f"  P1        : {test_at_feasible['precision_label1']:.4f}")
        print(f"  R1        : {test_at_feasible['recall_label1']:.4f}")
        print(f"  F1_1      : {test_at_feasible['f1_label1']:.4f}")
    else:
        feasible_th = None
        test_at_feasible = None
        print("\n[VAL feasible 0.85 threshold]")
        print("  Not found: no threshold satisfies P>=0.85 and R>=0.85 on validation set.")

    result = {
        "model_name": model_name,
        "val_search": val_search,
        "test_at_val_best_f1_threshold": test_at_val_best,
        "test_oracle_diagnostic_only": test_oracle,
        "test_at_val_feasible_085_threshold": test_at_feasible,
    }

    save_json(
        result,
        os.path.join(out_dir, f"{model_name}_flow_only_metrics.json"),
    )

    return result


# =============================================================================
# Flow-only MLP
# =============================================================================

class FlowOnlyDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y.astype(np.int64)).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class FlowOnlyMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.2,
        num_classes: int = 2,
    ):
        super().__init__()

        layers: List[nn.Module] = []
        in_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h

        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits


def parse_hidden_dims(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_torch_loss(
    args,
    train_labels: np.ndarray,
    device: torch.device,
) -> nn.Module:
    loss_name = args.loss.lower()

    class_counts = np.bincount(train_labels.astype(int), minlength=2)
    n0, n1 = int(class_counts[0]), int(class_counts[1])

    alpha = None

    if args.alpha_mode == "none":
        alpha = None

    elif args.alpha_mode == "mild":
        alpha = torch.tensor([1.0, 1.25], dtype=torch.float32, device=device)

    elif args.alpha_mode == "balanced":
        # mean-normalized inverse frequency
        weights = np.array([
            (n0 + n1) / (2.0 * max(n0, 1)),
            (n0 + n1) / (2.0 * max(n1, 1)),
        ], dtype=np.float32)
        weights = weights / weights.mean()
        alpha = torch.tensor(weights, dtype=torch.float32, device=device)

    elif args.alpha_mode == "custom":
        vals = [float(x) for x in args.alpha.split(",")]
        if len(vals) != 2:
            raise ValueError("--alpha must contain two values, e.g. 1.0,2.0")
        alpha = torch.tensor(vals, dtype=torch.float32, device=device)

    else:
        raise ValueError(f"Unknown alpha_mode: {args.alpha_mode}")

    print("\n[LOSS]")
    print(f"  loss       : {loss_name}")
    print(f"  alpha_mode : {args.alpha_mode}")
    print(f"  alpha      : {alpha.detach().cpu().tolist() if alpha is not None else None}")
    print(f"  counts     : class0={n0}, class1={n1}")

    if loss_name == "ce":
        criterion = nn.CrossEntropyLoss(weight=alpha)

    elif loss_name == "focal":
        criterion = FocalLossWithLabelSmoothing(
            alpha=alpha,
            gamma=float(args.focal_gamma),
            label_smoothing=float(args.label_smoothing),
        )

    elif loss_name == "cb_focal":
        criterion = ClassBalancedFocalLoss(
            labels=train_labels.tolist(),
            num_classes=2,
            beta=float(args.cb_beta),
            gamma=float(args.focal_gamma),
            label_smoothing=float(args.label_smoothing),
        )

    elif loss_name == "asymmetric":
        criterion = AsymmetricFocalLoss(
            gamma_pos=float(args.asl_gamma_pos),
            gamma_neg=float(args.asl_gamma_neg),
            alpha_pos=float(args.asl_alpha_pos),
        )

    elif loss_name == "hard_neg_ce":
        criterion = HardNegativeMiningCELoss(
            neg_keep_ratio=float(args.neg_keep_ratio),
            label_smoothing=float(args.label_smoothing),
        )

    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

    criterion = criterion.to(device)

    # 兼容你当前 ClassBalancedFocalLoss 中 alpha 不是 buffer 的写法
    if hasattr(criterion, "alpha") and isinstance(criterion.alpha, torch.Tensor):
        criterion.alpha = criterion.alpha.to(device)

    return criterion


def build_controlled_sampler(
    labels: np.ndarray,
    pos_fraction: Optional[float],
) -> Optional[WeightedRandomSampler]:
    if pos_fraction is None:
        return None

    labels = labels.astype(int)
    counts = np.bincount(labels, minlength=2)
    n0, n1 = int(counts[0]), int(counts[1])

    p = float(pos_fraction)
    if not 0.0 < p < 1.0:
        raise ValueError("pos_fraction must be between 0 and 1.")

    w0 = (1.0 - p) / max(n0, 1)
    w1 = p / max(n1, 1)

    sample_weights = np.where(labels == 1, w1, w0).astype(np.float64)

    print("\n[SAMPLER]")
    print(f"  target_pos_fraction : {p}")
    print(f"  class_counts        : class0={n0}, class1={n1}")
    print(f"  sample_weight_0     : {w0:.8e}")
    print(f"  sample_weight_1     : {w1:.8e}")

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def predict_mlp(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    probs = []

    loader = DataLoader(
        FlowOnlyDataset(x, np.zeros(len(x), dtype=np.int64)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.softmax(logits.float(), dim=-1)[:, 1]
            probs.append(prob.detach().cpu().numpy())

    return np.concatenate(probs, axis=0)


def evaluate_mlp_loss(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    criterion: nn.Module,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()

    loader = DataLoader(
        FlowOnlyDataset(x, y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    total_loss = 0.0
    total_n = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)

            loss = criterion(logits.float(), yb)

            bs = len(yb)
            total_loss += float(loss.item()) * bs
            total_n += bs

    return total_loss / max(total_n, 1)


def train_flow_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    args,
    out_dir: str,
    device: torch.device,
) -> Dict[str, Any]:
    print(f"\n{'#' * 80}")
    print("[MODEL] Flow-only MLP")
    print(f"{'#' * 80}")

    input_dim = x_train.shape[1]
    hidden_dims = parse_hidden_dims(args.mlp_hidden_dims)

    model = FlowOnlyMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=float(args.mlp_dropout),
        num_classes=2,
    ).to(device)

    criterion = build_torch_loss(args, y_train, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.mlp_lr),
        weight_decay=float(args.mlp_weight_decay),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    sampler = build_controlled_sampler(
        labels=y_train,
        pos_fraction=args.mlp_sampler_pos_fraction,
    )

    train_loader = DataLoader(
        FlowOnlyDataset(x_train, y_train),
        batch_size=int(args.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )

    best_score = -1e18
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, int(args.mlp_epochs) + 1):
        model.train()

        total_loss = 0.0
        used_batches = 0
        skipped_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = criterion(logits.float(), yb)

            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                skipped_batches += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item())
            used_batches += 1

        if used_batches == 0:
            raise RuntimeError("All MLP batches skipped due to NaN/Inf.")

        train_loss = total_loss / used_batches

        val_prob = predict_mlp(
            model=model,
            x=x_val,
            batch_size=int(args.batch_size),
            device=device,
        )

        val_loss = evaluate_mlp_loss(
            model=model,
            x=x_val,
            y=y_val,
            criterion=criterion,
            batch_size=int(args.batch_size),
            device=device,
        )

        val_search = threshold_search(y_val, val_prob)
        val_best = val_search["best_f1"]
        val_min_pr = val_search["best_min_pr"]

        if args.monitor == "val_f1_label1":
            score = float(val_best["f1_label1"])
        elif args.monitor == "val_min_pr_label1":
            score = float(val_min_pr["min_pr"])
        else:
            score = float(val_best["pr_auc"])

        scheduler.step(score)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_best_threshold": val_best["threshold"],
            "val_precision_label1": val_best["precision_label1"],
            "val_recall_label1": val_best["recall_label1"],
            "val_f1_label1": val_best["f1_label1"],
            "val_pr_auc": val_best["pr_auc"],
            "val_auc": val_best["auc"],
            "score": score,
            "skipped_batches": skipped_batches,
        }

        history.append(row)

        print(
            f"[MLP Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_P1={val_best['precision_label1']:.4f} "
            f"val_R1={val_best['recall_label1']:.4f} "
            f"val_F1_1={val_best['f1_label1']:.4f} "
            f"val_PR_AUC={val_best['pr_auc']:.4f} "
            f"score={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(args.mlp_patience):
                print(f"[MLP] early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("MLP did not produce a valid checkpoint.")

    model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "args": vars(args),
        },
        os.path.join(out_dir, "flow_only_mlp_best.pt"),
    )

    val_prob = predict_mlp(model, x_val, int(args.batch_size), device)
    test_prob = predict_mlp(model, x_test, int(args.batch_size), device)

    result = evaluate_prob_model(
        model_name="flow_only_mlp",
        y_val=y_val,
        val_prob1=val_prob,
        y_test=y_test,
        test_prob1=test_prob,
        out_dir=out_dir,
    )

    result["best_epoch"] = best_epoch
    result["best_score"] = best_score
    result["history"] = history

    save_json(result, os.path.join(out_dir, "flow_only_mlp_full_result.json"))

    return result


# =============================================================================
# XGBoost / LightGBM
# =============================================================================

def train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    args,
    out_dir: str,
) -> Optional[Dict[str, Any]]:
    try:
        import xgboost as xgb
    except Exception as e:
        print(f"[SKIP] xgboost is not available: {e}")
        return None

    print(f"\n{'#' * 80}")
    print("[MODEL] Flow-only XGBoost")
    print(f"{'#' * 80}")

    counts = np.bincount(y_train.astype(int), minlength=2)
    n0, n1 = int(counts[0]), int(counts[1])
    scale_pos_weight = n0 / max(n1, 1)

    print(f"[XGB] scale_pos_weight={scale_pos_weight:.4f}")

    model = xgb.XGBClassifier(
        n_estimators=int(args.xgb_n_estimators),
        max_depth=int(args.xgb_max_depth),
        learning_rate=float(args.xgb_lr),
        subsample=float(args.xgb_subsample),
        colsample_bytree=float(args.xgb_colsample_bytree),
        min_child_weight=float(args.xgb_min_child_weight),
        reg_lambda=float(args.xgb_reg_lambda),
        reg_alpha=float(args.xgb_reg_alpha),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method=args.xgb_tree_method,
        scale_pos_weight=scale_pos_weight,
        n_jobs=int(args.n_jobs),
        random_state=int(args.seed),
    )

    model.fit(x_train, y_train)

    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]

    result = evaluate_prob_model(
        model_name="flow_only_xgboost",
        y_val=y_val,
        val_prob1=val_prob,
        y_test=y_test,
        test_prob1=test_prob,
        out_dir=out_dir,
    )

    return result


def train_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    args,
    out_dir: str,
) -> Optional[Dict[str, Any]]:
    try:
        import lightgbm as lgb
    except Exception as e:
        print(f"[SKIP] lightgbm is not available: {e}")
        return None

    print(f"\n{'#' * 80}")
    print("[MODEL] Flow-only LightGBM")
    print(f"{'#' * 80}")

    counts = np.bincount(y_train.astype(int), minlength=2)
    n0, n1 = int(counts[0]), int(counts[1])
    scale_pos_weight = n0 / max(n1, 1)

    print(f"[LGBM] scale_pos_weight={scale_pos_weight:.4f}")

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=int(args.lgbm_n_estimators),
        learning_rate=float(args.lgbm_lr),
        num_leaves=int(args.lgbm_num_leaves),
        max_depth=int(args.lgbm_max_depth),
        min_child_samples=int(args.lgbm_min_child_samples),
        subsample=float(args.lgbm_subsample),
        colsample_bytree=float(args.lgbm_colsample_bytree),
        reg_lambda=float(args.lgbm_reg_lambda),
        reg_alpha=float(args.lgbm_reg_alpha),
        scale_pos_weight=scale_pos_weight,
        random_state=int(args.seed),
        n_jobs=int(args.n_jobs),
        verbosity=-1,
    )

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="binary_logloss",
    )

    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]

    result = evaluate_prob_model(
        model_name="flow_only_lightgbm",
        y_val=y_val,
        val_prob1=val_prob,
        y_test=y_test,
        test_prob1=test_prob,
        out_dir=out_dir,
    )

    return result


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--val_npz", required=True)
    parser.add_argument("--test_npz", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument(
        "--models",
        nargs="+",
        default=["mlp", "xgb", "lgbm"],
        choices=["mlp", "xgb", "lgbm"],
    )

    parser.add_argument("--seed", type=int, default=130)
    parser.add_argument("--n_jobs", type=int, default=-1)

    # MLP settings
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--mlp_epochs", type=int, default=150)
    parser.add_argument("--mlp_patience", type=int, default=20)
    parser.add_argument("--mlp_lr", type=float, default=1e-4)
    parser.add_argument("--mlp_weight_decay", type=float, default=5e-4)
    parser.add_argument("--mlp_dropout", type=float, default=0.20)
    parser.add_argument("--mlp_hidden_dims", type=str, default="256,128,64")

    # Controlled sampler.
    # 建议先跑 0.10 / 0.15 / 0.20。
    # 设置为 -1 表示不使用 sampler。
    parser.add_argument("--mlp_sampler_pos_fraction", type=float, default=-1.0)

    # Loss settings for MLP
    parser.add_argument(
        "--loss",
        type=str,
        default="cb_focal",
        choices=["ce", "focal", "cb_focal", "asymmetric", "hard_neg_ce"],
    )
    parser.add_argument(
        "--alpha_mode",
        type=str,
        default="none",
        choices=["none", "mild", "balanced", "custom"],
    )
    parser.add_argument("--alpha", type=str, default="1.0,2.0")

    parser.add_argument("--focal_gamma", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument("--cb_beta", type=float, default=0.9999)

    parser.add_argument("--asl_gamma_pos", type=float, default=0.0)
    parser.add_argument("--asl_gamma_neg", type=float, default=2.0)
    parser.add_argument("--asl_alpha_pos", type=float, default=0.55)

    parser.add_argument("--neg_keep_ratio", type=float, default=0.30)

    parser.add_argument(
        "--monitor",
        type=str,
        default="val_f1_label1",
        choices=["val_f1_label1", "val_min_pr_label1", "val_pr_auc"],
    )

    # XGBoost settings
    parser.add_argument("--xgb_n_estimators", type=int, default=800)
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_lr", type=float, default=0.03)
    parser.add_argument("--xgb_subsample", type=float, default=0.85)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.85)
    parser.add_argument("--xgb_min_child_weight", type=float, default=3.0)
    parser.add_argument("--xgb_reg_lambda", type=float, default=2.0)
    parser.add_argument("--xgb_reg_alpha", type=float, default=0.0)
    parser.add_argument("--xgb_tree_method", type=str, default="hist")

    # LightGBM settings
    parser.add_argument("--lgbm_n_estimators", type=int, default=1000)
    parser.add_argument("--lgbm_lr", type=float, default=0.03)
    parser.add_argument("--lgbm_num_leaves", type=int, default=63)
    parser.add_argument("--lgbm_max_depth", type=int, default=-1)
    parser.add_argument("--lgbm_min_child_samples", type=int, default=30)
    parser.add_argument("--lgbm_subsample", type=float, default=0.85)
    parser.add_argument("--lgbm_colsample_bytree", type=float, default=0.85)
    parser.add_argument("--lgbm_reg_lambda", type=float, default=2.0)
    parser.add_argument("--lgbm_reg_alpha", type=float, default=0.0)

    args = parser.parse_args()

    if args.mlp_sampler_pos_fraction < 0:
        args.mlp_sampler_pos_fraction = None

    return args


def main():
    args = parse_args()

    set_seed(args.seed)
    safe_mkdir(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    x_train, y_train = load_flow_npz(args.train_npz)
    x_val, y_val = load_flow_npz(args.val_npz)
    x_test, y_test = load_flow_npz(args.test_npz)

    print_dataset_report("train", x_train, y_train)
    print_dataset_report("val", x_val, y_val)
    print_dataset_report("test", x_test, y_test)

    results: Dict[str, Any] = {}

    if "mlp" in args.models:
        results["mlp"] = train_flow_mlp(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            args=args,
            out_dir=args.out_dir,
            device=device,
        )

    if "xgb" in args.models:
        results["xgb"] = train_xgboost(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            args=args,
            out_dir=args.out_dir,
        )

    if "lgbm" in args.models:
        results["lgbm"] = train_lightgbm(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            args=args,
            out_dir=args.out_dir,
        )

    save_json(
        {
            "args": vars(args),
            "results": results,
        },
        os.path.join(args.out_dir, "flow_only_all_results.json"),
    )

    print("\n" + "=" * 80)
    print("[SUMMARY] Test at validation-selected threshold")
    print("=" * 80)

    for name, result in results.items():
        if result is None:
            continue

        m = result["test_at_val_best_f1_threshold"]

        print(
            f"{name:>8s} | "
            f"P1={m['precision_label1']:.4f} | "
            f"R1={m['recall_label1']:.4f} | "
            f"F1_1={m['f1_label1']:.4f} | "
            f"PR_AUC={m['pr_auc']:.4f} | "
            f"AUC={m['auc']:.4f} | "
            f"threshold={m['threshold']:.4f}"
        )

    print("\n[INFO] Saved results to:", args.out_dir)


if __name__ == "__main__":
    main()