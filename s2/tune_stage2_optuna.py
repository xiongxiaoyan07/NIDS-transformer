from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from stage2.utils import worker_init_fn
from stage2.config import load_config
from stage2.context import ContextIndexBuilder
from stage2.data_io import prepare_sorted_stage2_data
from stage2.losses import FocalLossWithLabelSmoothing, compute_class_alpha
from stage2.metrics import classification_metrics
from stage2.model import build_stage2_model


# ============================================================
# Reproducibility and runtime helpers
# ============================================================

def seed_everything(seed: int) -> None:
    """Fast, reproducible-enough setup for Colab parameter search."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Parameter search values speed over bit-for-bit reproducibility.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(False)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_grad_scaler(enabled: bool):
    """Works with both old and new PyTorch AMP APIs."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ============================================================
# Dataset that shares one Stage1 embedding array across splits
# ============================================================

class SharedStage2Dataset(Dataset):
    def __init__(
        self,
        meta_df_sorted: pd.DataFrame,
        z_sorted: np.ndarray,
        context_indices: List[np.ndarray],
        target_split: str,
    ) -> None:
        self.z = np.ascontiguousarray(z_sorted, dtype=np.float32)
        self.context_indices = context_indices

        split_values = meta_df_sorted["split"].astype(str).to_numpy()
        labels_all = meta_df_sorted["label"].to_numpy(dtype=np.int64, copy=False)
        flow_ids_all = meta_df_sorted["flow_id"].to_numpy(dtype=np.int64, copy=False)

        self.target_rows = np.flatnonzero(split_values == target_split).astype(np.int64)
        if self.target_rows.size == 0:
            raise ValueError(f"No rows found for split={target_split}")

        self.labels = labels_all[self.target_rows]
        self.flow_ids = flow_ids_all[self.target_rows]

    def __len__(self) -> int:
        return int(self.target_rows.size)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        row_idx = int(self.target_rows[i])
        ctx_idx = self.context_indices[row_idx]

        if len(ctx_idx) == 0:
            context_z = torch.empty((0, self.z.shape[1]), dtype=torch.float32)
        else:
            # NumPy advanced indexing returns a compact array for this sample only.
            context_z = torch.from_numpy(self.z[ctx_idx])

        return {
            "context_z": context_z,
            "label": int(self.labels[i]),
            "flow_id": int(self.flow_ids[i]),
            "row_idx": row_idx,
        }


def stage2_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    input_dim = int(batch[0]["context_z"].shape[1])
    max_len = max(1, max(int(item["context_z"].shape[0]) for item in batch))

    x = torch.zeros((batch_size, max_len, input_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        length = int(item["context_z"].shape[0])
        if length > 0:
            # Keep the same left-padding convention as the user's Stage2 code.
            x[i, max_len - length :] = item["context_z"]
            mask[i, max_len - length :] = True

    return {
        "context_z": x,
        "mask": mask,
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "flow_id": torch.tensor([item["flow_id"] for item in batch], dtype=torch.long),
        "row_idx": torch.tensor([item["row_idx"] for item in batch], dtype=torch.long),
    }


# ============================================================
# Context cache
# ============================================================

class ContextCache:
    """Small LRU cache so Colab RAM is not filled with many context copies."""

    def __init__(
        self,
        meta_df: pd.DataFrame,
        base_cfg: Dict[str, Any],
        max_items: int = 1,
    ) -> None:
        self.meta_df = meta_df
        self.base_cfg = base_cfg
        self.max_items = max(1, int(max_items))
        self._cache: OrderedDict[int, List[np.ndarray]] = OrderedDict()

    def get(self, window_size: int) -> List[np.ndarray]:
        window_size = int(window_size)
        if window_size in self._cache:
            contexts = self._cache.pop(window_size)
            self._cache[window_size] = contexts
            return contexts

        cfg = copy.deepcopy(self.base_cfg)
        cfg["context"]["window_size"] = window_size
        print(f"[TUNE] Building exact context indices for window_size={window_size}")
        contexts = ContextIndexBuilder(self.meta_df, cfg).build()

        self._cache[window_size] = contexts
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
            cleanup_cuda()
        return contexts


# ============================================================
# Loss, loaders, optimizer
# ============================================================

def make_loss_fn(
    train_labels: np.ndarray,
    cfg: Dict[str, Any],
    device: torch.device,
) -> nn.Module:
    train_cfg = cfg["training"]
    loss_type = str(train_cfg.get("loss_type", "focal"))
    weighted = bool(train_cfg.get("class_weighted_loss", True))

    if loss_type == "ce":
        if not weighted:
            return nn.CrossEntropyLoss()

        counts = np.bincount(train_labels.astype(int), minlength=2).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        weights = counts.sum() / (len(counts) * counts)
        return nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device)
        )

    if loss_type == "focal":
        alpha = compute_class_alpha(train_labels, num_classes=2).to(device) if weighted else None
        return FocalLossWithLabelSmoothing(
            alpha=alpha,
            gamma=float(train_cfg.get("focal_gamma", 2.0)),
            label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
        )

    raise ValueError(f"Unknown training.loss_type: {loss_type}")


def make_loader(
    dataset: SharedStage2Dataset,
    cfg: Dict[str, Any],
    train: bool,
    seed: int,
) -> DataLoader:
    train_cfg = cfg["training"]
    batch_size = int(
        train_cfg.get("batch_size", 128)
        if train
        else train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 128))
    )
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = torch.cuda.is_available()

    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": stage2_collate_fn,
        "worker_init_fn": worker_init_fn,
        "pin_memory": pin_memory,
        "generator": torch.Generator().manual_seed(seed),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = False
        kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))

    use_sampler = train and bool(train_cfg.get("use_weighted_sampler", False))
    if use_sampler:
        labels = dataset.labels.astype(int)
        counts = np.maximum(np.bincount(labels, minlength=2).astype(np.float64), 1.0)
        sample_weights = (1.0 / counts)[labels]
        kwargs["sampler"] = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        kwargs["shuffle"] = bool(train)

    return DataLoader(**kwargs)


def make_optimizer(model: nn.Module, cfg: Dict[str, Any], device: torch.device):
    kwargs: Dict[str, Any] = {
        "lr": float(cfg["training"].get("lr", 3e-4)),
        "weight_decay": float(cfg["training"].get("weight_decay", 1e-4)),
    }
    if device.type == "cuda":
        kwargs["fused"] = True

    try:
        return torch.optim.AdamW(model.parameters(), **kwargs)
    except (TypeError, RuntimeError):
        kwargs.pop("fused", None)
        return torch.optim.AdamW(model.parameters(), **kwargs)


# ============================================================
# Train and validation loops
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler,
    amp_enabled: bool,
    grad_clip_norm: Optional[float],
) -> float:
    model.train()
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_n = 0

    for batch in loader:
        x = batch["context_z"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(x, mask)
            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()

        # Important: unscale before gradient clipping.
        if grad_clip_norm is not None and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        bs = int(y.size(0))
        total_loss += loss.detach().float() * bs
        total_n += bs

    return float((total_loss / max(total_n, 1)).item())


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    model.eval()

    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_n = 0
    all_y: List[torch.Tensor] = []
    all_score: List[torch.Tensor] = []

    for batch in loader:
        x = batch["context_z"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(x, mask)
            loss = loss_fn(logits, y)
            score = torch.softmax(logits, dim=-1)[:, 1]

        bs = int(y.size(0))
        total_loss += loss.detach().float() * bs
        total_n += bs
        all_y.append(y.detach())
        all_score.append(score.detach().float())

    y_true = torch.cat(all_y).cpu().numpy()
    y_score = torch.cat(all_score).cpu().numpy()
    y_pred = (y_score >= float(threshold)).astype(np.int64)
    avg_loss = float((total_loss / max(total_n, 1)).item())

    return classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        loss=avg_loss,
        threshold=threshold,
        num_classes=2,
    )


def objective_value(metrics: Dict[str, Any], metric_name: str) -> float:
    if metric_name == "loss":
        return -float(metrics["loss"])
    value = metrics.get(metric_name)
    if value is None or not np.isfinite(float(value)):
        return -float("inf")
    return float(value)


# ============================================================
# Search space
# ============================================================

def parse_int_list(text: str) -> List[int]:
    if not text.strip():
        return []
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    return sorted(set(values))


def d_model_choices(input_dim: int, profile: str) -> List[str]:
    if profile == "fast":
        raw = [64, 128, 256, input_dim]
    else:
        raw = [64, 96, 128, 192, 256, 384, 512, input_dim]

    # Compression-only search is safer and faster on Colab.
    dims = sorted({int(d) for d in raw if 16 <= int(d) <= int(input_dim)})
    if int(input_dim) not in dims:
        dims.append(int(input_dim))
    return [str(d) for d in sorted(set(dims))]


def resolve_config_from_params(
    base_cfg: Dict[str, Any],
    params: Dict[str, Any],
    input_dim: int,
    args: argparse.Namespace,
    for_final_training: bool,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)

    cfg["context"]["window_size"] = int(
        params.get("window_size", base_cfg["context"].get("window_size", 64))
    )

    d_model = int(params["d_model"])
    cfg["model"].update(
        {
            "model_type": "transformer",
            "d_model": d_model,
            "nhead": int(params["nhead"]),
            "num_layers": int(params["num_layers"]),
            "dim_feedforward": int(d_model * int(params["ff_multiplier"])),
            "dropout": float(params["dropout"]),
            "pooling": str(params["pooling"]),
            "use_positional_encoding": bool(params["use_positional_encoding"]),
            "cls_head": int(params["cls_head"]),
        }
    )

    cfg["training"].update(
        {
            "epochs": int(args.final_epochs if for_final_training else args.max_epochs),
            "patience": int(args.final_patience if for_final_training else args.patience),
            "batch_size": int(params["batch_size"]),
            "eval_batch_size": int(params.get("eval_batch_size", params["batch_size"])),
            "lr": float(params["lr"]),
            "weight_decay": float(params["weight_decay"]),
            "grad_clip_norm": float(params["grad_clip_norm"]),
            "loss_type": str(params["loss_type"]),
            "class_weighted_loss": True,
            "use_weighted_sampler": False,
            "metric_for_best": str(args.objective_metric),
            "threshold": 0.5,
            "auto_threshold": True,
            "num_workers": int(args.num_workers),
            "device": "auto",
            "min_delta": float(args.min_delta),
            "amp": True,
        }
    )

    if params["loss_type"] == "focal":
        cfg["training"]["focal_gamma"] = float(params["focal_gamma"])
        cfg["training"]["label_smoothing"] = float(params["label_smoothing"])

    return cfg


def sample_trial_config(
    trial: optuna.Trial,
    base_cfg: Dict[str, Any],
    input_dim: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    dims = d_model_choices(input_dim, args.profile)
    d_model = int(trial.suggest_categorical("d_model", dims))

    head_candidates = [h for h in [1, 2, 4, 8] if h <= d_model and d_model % h == 0]
    if args.profile == "fast":
        head_candidates = [h for h in head_candidates if h in {2, 4, 8}] or head_candidates

    if args.window_sizes:
        trial.suggest_categorical("window_size", args.window_sizes)

    trial.suggest_categorical("nhead", head_candidates)
    trial.suggest_categorical("num_layers", [1, 2] if args.profile == "fast" else [1, 2, 3])
    trial.suggest_categorical("ff_multiplier", [2, 4])
    trial.suggest_float("dropout", 0.10, 0.40, step=0.05)
    trial.suggest_categorical(
        "pooling",
        ["last", "mean"] if args.profile == "fast" else ["last", "mean", "attention"],
    )
    trial.suggest_categorical("use_positional_encoding", [False, True])
    trial.suggest_categorical("cls_head", [1, 2])

    trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    trial.suggest_categorical("batch_size", args.batch_sizes)
    trial.suggest_categorical("eval_batch_size", args.eval_batch_sizes)
    trial.suggest_categorical("grad_clip_norm", [0.5, 1.0, 2.0])

    loss_type = trial.suggest_categorical("loss_type", ["focal", "ce"])
    if loss_type == "focal":
        trial.suggest_float("focal_gamma", 1.0, 3.0, step=0.5)
        trial.suggest_categorical("label_smoothing", [0.0, 0.02, 0.05, 0.1])

    params = dict(trial.params)
    # If window_size is not searched, carry the YAML value explicitly.
    params.setdefault("window_size", int(base_cfg["context"].get("window_size", 64)))

    return resolve_config_from_params(
        base_cfg=base_cfg,
        params=params,
        input_dim=input_dim,
        args=args,
        for_final_training=False,
    )


# ============================================================
# One Optuna trial
# ============================================================

def run_trial(
    trial: optuna.Trial,
    base_cfg: Dict[str, Any],
    meta_df: pd.DataFrame,
    z_sorted: np.ndarray,
    context_cache: ContextCache,
    input_dim: int,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    seed_everything(args.seed)
    cfg = sample_trial_config(trial, base_cfg, input_dim, args)
    window_size = int(cfg["context"]["window_size"])
    context_indices = context_cache.get(window_size)

    datasets = {
        split: SharedStage2Dataset(
            meta_df_sorted=meta_df,
            z_sorted=z_sorted,
            context_indices=context_indices,
            target_split=split,
        )
        for split in ["train", "val"]
    }

    train_loader = make_loader(datasets["train"], cfg, train=True, seed=args.seed)
    val_loader = make_loader(datasets["val"], cfg, train=False, seed=args.seed)

    model: Optional[nn.Module] = None
    optimizer = None
    loss_fn = None

    try:
        model = build_stage2_model(cfg, input_dim=input_dim).to(device)
        loss_fn = make_loss_fn(datasets["train"].labels, cfg, device)
        optimizer = make_optimizer(model, cfg, device)

        max_epochs = int(cfg["training"]["epochs"])
        warmup_epochs = max(1, min(3, max_epochs // 10))
        cosine_epochs = max(1, max_epochs - warmup_epochs)
        lr = float(cfg["training"]["lr"])

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=lr * 0.01,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

        amp_enabled = device.type == "cuda"
        scaler = make_grad_scaler(enabled=amp_enabled)
        grad_clip_norm = float(cfg["training"].get("grad_clip_norm", 1.0))

        best_score = -float("inf")
        best_epoch = 0
        best_metrics: Dict[str, Any] = {}
        bad_epochs = 0
        started = time.time()

        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                scaler=scaler,
                amp_enabled=amp_enabled,
                grad_clip_norm=grad_clip_norm,
            )
            scheduler.step()

            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                amp_enabled=amp_enabled,
                threshold=0.5,
            )
            score = objective_value(val_metrics, args.objective_metric)

            trial.report(score, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned at epoch={epoch}, score={score:.6f}")

            if score > best_score + float(args.min_delta):
                best_score = score
                best_epoch = epoch
                best_metrics = val_metrics
                bad_epochs = 0
            else:
                bad_epochs += 1

            print(
                f"[TRIAL {trial.number:03d}][EPOCH {epoch:03d}] "
                f"train_loss={train_loss:.6f} "
                f"val_macro_f1={val_metrics['macro_f1']:.5f} "
                f"val_f1_label1={val_metrics['f1_label1']:.5f} "
                f"val_auc={val_metrics['auc']:.5f} "
                f"objective={score:.5f}"
            )

            if bad_epochs >= int(args.patience):
                break

        elapsed = time.time() - started
        trial.set_user_attr("best_epoch", int(best_epoch))
        trial.set_user_attr("elapsed_seconds", float(elapsed))
        trial.set_user_attr("val_macro_f1", float(best_metrics.get("macro_f1", 0.0)))
        trial.set_user_attr("val_f1_label1", float(best_metrics.get("f1_label1", 0.0)))
        trial.set_user_attr("val_auc", float(best_metrics.get("auc", 0.0)))
        trial.set_user_attr("val_pr_auc", float(best_metrics.get("pr_auc", 0.0)))
        trial.set_user_attr("val_loss", float(best_metrics.get("loss", float("inf"))))
        return float(best_score)

    except torch.cuda.OutOfMemoryError as exc:
        print(f"[TRIAL {trial.number:03d}] CUDA OOM; pruning this parameter set.")
        raise optuna.TrialPruned("CUDA out of memory") from exc
    finally:
        del train_loader, val_loader, datasets
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
        if loss_fn is not None:
            del loss_fn
        cleanup_cuda()


# ============================================================
# Output helpers
# ============================================================

def save_json(data: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_best_config(
    study: optuna.Study,
    base_cfg: Dict[str, Any],
    input_dim: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    params = dict(study.best_trial.params)
    params.setdefault("window_size", int(base_cfg["context"].get("window_size", 64)))
    return resolve_config_from_params(
        base_cfg=base_cfg,
        params=params,
        input_dim=input_dim,
        args=args,
        for_final_training=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for the user's Stage2 Transformer on Colab"
    )
    parser.add_argument("--stage1_dir", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)

    parser.add_argument("--study_name", default="stage2_transformer_search")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Maximum search time in seconds; 0 means no time limit.",
    )
    parser.add_argument("--profile", choices=["fast", "balanced"], default="fast")
    parser.add_argument(
        "--objective_metric",
        choices=["macro_f1", "f1_label1", "auc", "pr_auc", "loss"],
        default="macro_f1",
    )

    parser.add_argument("--max_epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min_delta", type=float, default=5e-4)
    parser.add_argument("--final_epochs", type=int, default=150)
    parser.add_argument("--final_patience", type=int, default=15)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="64,128,256",
        help="Comma-separated training batch sizes.",
    )
    parser.add_argument(
        "--eval_batch_sizes",
        type=str,
        default="128,256,512",
        help="Comma-separated validation batch sizes.",
    )
    parser.add_argument(
        "--window_sizes",
        type=str,
        default="",
        help=(
            "Optional comma-separated context windows. Empty keeps the YAML value fixed, "
            "which is recommended for the first Colab search."
        ),
    )
    parser.add_argument("--context_cache_size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.batch_sizes = parse_int_list(args.batch_sizes)
    args.eval_batch_sizes = parse_int_list(args.eval_batch_sizes)
    args.window_sizes = parse_int_list(args.window_sizes)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(args.config)
    base_cfg["data"]["stage1_dir"] = args.stage1_dir
    base_cfg["seed"] = int(args.seed)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TUNE] device={device}")
    if device.type != "cuda":
        print("[WARNING] CUDA is not available. In Colab select Runtime > Change runtime type > GPU.")

    # Important: Stage1 data is loaded once for all trials.
    print("[TUNE] Loading and sorting Stage1 outputs once...")
    meta_df, z_sorted = prepare_sorted_stage2_data(args.stage1_dir, base_cfg)
    input_dim = int(z_sorted.shape[1])
    print(f"[TUNE] flows={len(meta_df)}, input_dim={input_dim}")
    print(f"[TUNE] split counts={meta_df['split'].value_counts().to_dict()}")

    context_cache = ContextCache(
        meta_df=meta_df,
        base_cfg=base_cfg,
        max_items=args.context_cache_size,
    )

    storage_path = out_dir / "stage2_optuna.db"
    storage = f"sqlite:///{storage_path}"
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=min(5, max(1, args.n_trials // 4)),
        n_warmup_steps=5,
        interval_steps=1,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        return run_trial(
            trial=trial,
            base_cfg=base_cfg,
            meta_df=meta_df,
            z_sorted=z_sorted,
            context_cache=context_cache,
            input_dim=input_dim,
            device=device,
            args=args,
        )

    timeout: Optional[int] = None if args.timeout <= 0 else int(args.timeout)
    study.optimize(
        objective,
        n_trials=int(args.n_trials),
        timeout=timeout,
        n_jobs=1,  # A single Colab GPU must not run concurrent trials.
        gc_after_trial=True,
        show_progress_bar=True,
    )

    trials_csv = out_dir / "optuna_trials.csv"
    study.trials_dataframe().to_csv(trials_csv, index=False)

    best_config = build_best_config(study, base_cfg, input_dim, args)
    best_config_path = out_dir / "best_stage2_config.yaml"
    with best_config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(best_config, f, sort_keys=False, allow_unicode=True)

    best_summary = {
        "study_name": study.study_name,
        "objective_metric": args.objective_metric,
        "best_trial": int(study.best_trial.number),
        "best_value": float(study.best_value),
        "best_params": dict(study.best_trial.params),
        "best_user_attrs": dict(study.best_trial.user_attrs),
        "storage": str(storage_path),
        "best_config": str(best_config_path),
    }
    save_json(best_summary, out_dir / "best_trial_summary.json")

    print("\n" + "=" * 72)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best {args.objective_metric}: {study.best_value:.6f}")
    print(json.dumps(study.best_trial.params, indent=2, ensure_ascii=False))
    print(f"Trials CSV: {trials_csv}")
    print(f"Best YAML:  {best_config_path}")
    print(f"Study DB:   {storage_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

# !pip -q install optuna


# !python /content/drive/MyDrive/s2/tune_stage2_optuna.py \
#   --stage1_dir /content/drive/MyDrive/s1/tensors_ar_002 \
#   --config /content/drive/MyDrive/s2/stage2_config_0619.yaml \
#   --out_dir /content/drive/MyDrive/s2/stage2_optuna_0619 \
#   --study_name stage2_transformer_v1 \
#   --n_trials 30 \
#   --max_epochs 35 \
#   --patience 6 \
#   --final_epochs 150 \
#   --final_patience 15 \
#   --objective_metric macro_f1 \
#   --profile fast \
#   --num_workers 2 \
#   --batch_sizes 64,128,256 \
#   --eval_batch_sizes 128,256,512
