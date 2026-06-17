from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import Stage2Dataset, stage2_collate_fn
from .losses import compute_class_alpha,FocalLossWithLabelSmoothing
from .utils import binary_metrics, metric_value, save_json
from .metrics import classification_metrics
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

def make_loss_fn(train_labels: np.ndarray, cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    train_cfg = cfg["training"]
    loss_type = train_cfg.get("loss_type", "focal")
    print("[INFO]---trainer.py---make_loss_fn---label 分布: {}".format(np.bincount(train_labels.astype(int), minlength=2).astype(np.float64)))
    if loss_type == "ce":
        if not bool(train_cfg.get("class_weighted_loss", True)):
            return nn.CrossEntropyLoss()

        counts = np.bincount(train_labels.astype(int), minlength=2).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        weights = counts.sum() / (len(counts) * counts)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

        print("[INFO] CE class counts:", counts.tolist())
        print("[INFO] CE class weights:", weights.tolist())

        return nn.CrossEntropyLoss(weight=weights_t)

    if loss_type == "focal":
        alpha = None
        if bool(train_cfg.get("class_weighted_loss", True)):
            alpha = compute_class_alpha(train_labels, num_classes=2).to(device)
        else:
            # ⭐ 手动指定 alpha（在配置中设置）
            manual_alpha = train_cfg.get("alpha", None)
            if manual_alpha is not None:
                alpha = torch.tensor(manual_alpha, dtype=torch.float32, device=device)
                print(f"[INFO] Using manual alpha: {manual_alpha}")

        gamma = float(train_cfg.get("focal_gamma", 2.0))
        label_smoothing = float(train_cfg.get("label_smoothing", 0.0))

        print(f"[INFO] Using FocalLoss: gamma={gamma}, label_smoothing={label_smoothing}")

        return FocalLossWithLabelSmoothing(
            alpha=alpha,
            gamma=gamma,
            label_smoothing=label_smoothing,
        )

    raise ValueError(f"Unknown training.loss_type: {loss_type}")

def make_train_loader(dataset: Stage2Dataset, cfg: Dict[str, Any]) -> DataLoader:
    train_cfg = cfg["training"]
    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 0))

    if bool(train_cfg.get("use_weighted_sampler", True)):
        labels = dataset.labels.astype(int)
        counts = np.bincount(labels, minlength=2).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        class_weights = 1.0 / counts
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        print(f"[INFO] weighted sampler class_counts: {counts.astype(int).tolist()}")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=stage2_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=stage2_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

def make_eval_loader(dataset: Stage2Dataset, cfg: Dict[str, Any]) -> DataLoader:
    train_cfg = cfg["training"]
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=stage2_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip_norm: Optional[float],
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        x = batch["context_z"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, mask)
        loss = loss_fn(logits, y)
        loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(total_n, 1)

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    threshold: Optional[float] = 0.5,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    model.eval()

    losses: List[float] = []
    all_y: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_score: List[np.ndarray] = []
    all_flow_ids: List[int] = []
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        x = batch["context_z"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with torch.no_grad():
            logits = model(x, mask)
            loss = loss_fn(logits, y)

            probs = torch.softmax(logits, dim=-1)
            score_label1 = probs[:, 1]

            # 如果提供了阈值，使用自定义阈值
            if threshold is None:
                preds = logits.argmax(dim=-1)
            else:
                preds = (score_label1 >= float(threshold)).long()

        losses.append(float(loss.item()) * y.size(0))
        all_y.append(y.detach().cpu().numpy())
        all_pred.append(preds.detach().cpu().numpy())
        all_score.append(score_label1.detach().cpu().numpy())
        all_flow_ids.extend(batch["flow_id"].cpu().numpy().tolist())
        total_loss += loss.item()
        num_batches += 1

    y_true = np.concatenate(all_y, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    y_score = np.concatenate(all_score, axis=0)
    avg_loss = total_loss / num_batches

    metrics = classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        num_classes=2,
        loss=avg_loss,
        threshold=threshold,
    )

    metrics["loss"] = avg_loss #float(np.sum(losses) / max(len(y_true), 1))
    metrics["num_samples"] = int(len(y_true))
    metrics["positive_count"] = int(y_true.sum())
    metrics["positive_rate"] = float(y_true.mean()) if len(y_true) else 0.0

    pred_df = pd.DataFrame(
        {
            "flow_id": np.array(all_flow_ids, dtype=np.int64),
            "label": y_true.astype(int),
            "prob_label_1": y_score.astype(float),
            "pred": y_pred.astype(int),
        }
    )

    return metrics, pred_df

def find_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "f1_label1",
    threshold_min: float = 0.01,
    threshold_max: float = 0.99,
    threshold_steps: int = 199,
) -> Tuple[float, Dict[str, Any]]:
    thresholds = np.linspace(threshold_min, threshold_max, threshold_steps)

    best_threshold = 0.5
    best_value = -float("inf")
    best_metrics: Dict[str, Any] = {}

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)

        metrics = classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            num_classes=2,
            threshold=float(t),
        )

        if metric == "f1_label1":
            value = metrics["f1_label1"]
        elif metric == "recall":
            value = metrics["recall_label1"]
        elif metric == "precision":
            value = metrics["precision_label1"]
        else:
            raise ValueError(f"Unknown threshold_metric: {metric}")

        if value > best_value:
            best_value = value
            best_threshold = float(t)
            best_metrics = metrics

    return best_threshold, best_metrics

class Stage2Trainer:
    def __init__(
            self,
            model: nn.Module,
            datasets: Dict[str, Stage2Dataset],
            cfg: Dict[str, Any],
            device: torch.device,
            out_dir: str,
            input_dim: int,
    ):
        self.model = model
        self.datasets = datasets
        self.cfg = cfg
        self.device = device
        self.out_dir = out_dir
        self.input_dim = input_dim

        self.train_loader = make_train_loader(datasets["train"], cfg)
        self.eval_loaders = {
            "train": make_eval_loader(datasets["train"], cfg),
            "val": make_eval_loader(datasets["val"], cfg),
            "test": make_eval_loader(datasets["test"], cfg),
        }

        self.loss_fn = make_loss_fn(datasets["train"].labels, cfg, device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg["training"].get("lr", 3e-4)),
            weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
        )
    # train model
    def fit(self) -> Dict[str, Any]:
        epochs = int(self.cfg["training"].get("epochs", 100))
        patience = int(self.cfg["training"].get("patience", 20))
        threshold = float(self.cfg["training"].get("threshold", 0.5))
        metric_for_best = self.cfg["training"].get("metric_for_best", "val_f1")
        print("[Stage2Trainer]---fit----metric_for_best = ", metric_for_best)

        grad_clip_norm = self.cfg["training"].get("grad_clip_norm", 1.0)
        grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)

        # 你的原代码 warmup_epochs = min(10, epochs // 10)。
        # 这里加 max(1, ...) 防止 epochs 较小时 warmup_epochs=0。
        warmup_epochs = max(1, min(10, epochs // 10))
        print("[Stage2Trainer] ---- fit ------warmup_epochs = ", warmup_epochs)
        cosine_epochs = max(1, epochs - warmup_epochs)

        lr = float(self.cfg["training"].get("lr", 3e-4))
        print("[Stage2Trainer]---fit lr= ", lr)
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_epochs,
            eta_min=lr * 0.01,
        )

        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        best_score = -float("inf")
        best_epoch = -1
        bad_epochs = 0
        epoch_rows: List[Dict[str, Any]] = []

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                model=self.model,
                loader=self.train_loader,
                optimizer=self.optimizer,
                loss_fn=self.loss_fn,
                device=self.device,
                grad_clip_norm=grad_clip_norm,
            )

            scheduler.step()

            metrics_by_split: Dict[str, Dict[str, Any]] = {}
            for split, loader in self.eval_loaders.items():
                metrics, _ = evaluate(
                    model=self.model,
                    loader=loader,
                    loss_fn=self.loss_fn,
                    device=self.device,
                    threshold=threshold,
                )
                metrics_by_split[split] = metrics

            score = metric_value(metrics_by_split, metric_for_best)
            improved = score > best_score
            if improved:
                best_score = score
                best_epoch = epoch
                bad_epochs = 0
                self._save_checkpoint(
                    path=os.path.join(self.out_dir, "stage2_best_model.pt"),
                    epoch=epoch,
                    best_score=best_score,
                    metric_for_best=metric_for_best,
                )
            else:
                bad_epochs += 1

            row = {
                "epoch": epoch,
                "train_loss_optim": train_loss,
                "best_epoch": best_epoch,
                "best_score": best_score,
            }
            for split, metrics in metrics_by_split.items():
                for key, value in metrics.items():
                    if key == "confusion_matrix":
                        continue
                    row[f"{split}_{key}"] = value
            epoch_rows.append(row)

            pd.DataFrame(epoch_rows).to_csv(
                os.path.join(self.out_dir, "stage2_epoch_metrics.csv"),
                index=False,
            )
            print(
                f"[EPOCH {epoch:03d}] "
                f"train_loss={train_loss:.6f} "
                f"val_loss={metrics_by_split['val']['loss']:.6f} "
                f"val_macro_f1={metrics_by_split['val']['macro_f1']:.4f} "
                f"val_weighted_f1={metrics_by_split['val']['weighted_f1']} "
                f"val_f1_label1={metrics_by_split['val']['f1_label1']:.4f} "
                f"val_auc={metrics_by_split['val']['auc']:.4f}"
                f"{'*' if improved else ''}"
            )

            if 0 < patience <= bad_epochs:
                print(f"[INFO] Early stopping at epoch={epoch}, best_epoch={best_epoch}")
                break
    # load best model
    def load_best(self) -> Dict[str, Any]:
        path = os.path.join(self.out_dir, "stage2_best_model.pt")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        return ckpt
    # evaluate the model
    def final_evaluate_and_save(self, meta_df, context_indices) -> Dict[str, Any]:
        threshold = float(self.cfg["training"].get("threshold", 0.5))
        ckpt = self.load_best()

        if bool(self.cfg["training"].get("auto_threshold", True)):
            _, val_pred_df = evaluate(
                model=self.model,
                loader=self.eval_loaders["val"],
                loss_fn=self.loss_fn,
                device=self.device,
                threshold=0.5,
            )

            threshold, threshold_metrics = find_best_threshold(
                y_true=val_pred_df["label"].to_numpy(),
                y_score=val_pred_df["prob_label_1"].to_numpy(),
                metric=self.cfg["training"].get("threshold_metric", "f1"),
                threshold_min=float(self.cfg["training"].get("threshold_min", 0.01)),
                threshold_max=float(self.cfg["training"].get("threshold_max", 0.99)),
                threshold_steps=int(self.cfg["training"].get("threshold_steps", 99)),
            )

            self.cfg["training"]["threshold"] = threshold
            print(f"[INFO] Auto-selected threshold on val: {threshold:.4f}")
            print(f"[INFO] Val metrics at selected threshold: {threshold_metrics}")

        context_lengths = np.array([len(x) for x in context_indices], dtype=np.int64)
        final_metrics: Dict[str, Any] = {
            "best_epoch": int(ckpt["epoch"]),
            "best_score": float(ckpt["best_score"]),
            "metric_for_best": ckpt["metric_for_best"],
            "threshold": float(threshold),
            "config": self.cfg,
            "data": {
                "num_flows": int(len(meta_df)),
                "z_dim": int(self.input_dim),
                "split_counts": {
                    str(k): int(v) for k, v in meta_df["split"].value_counts().to_dict().items()
                },
                "label_counts_by_split": {
                    str(split): {
                        str(k): int(v)
                        for k, v in group["label"].value_counts().sort_index().to_dict().items()
                    }
                    for split, group in meta_df.groupby("split")
                },
                "context_length": {
                    "min": int(context_lengths.min()),
                    "max": int(context_lengths.max()),
                    "mean": float(context_lengths.mean()),
                    "p50": float(np.percentile(context_lengths, 50)),
                    "p95": float(np.percentile(context_lengths, 95)),
                },
            },
            "splits": {},
        }

        for split, loader in self.eval_loaders.items():
            metrics, pred_df = evaluate(
                model=self.model,
                loader=loader,
                loss_fn=self.loss_fn,
                device=self.device,
                threshold=threshold,
            )
            final_metrics["splits"][split] = metrics

            pred_path = os.path.join(self.out_dir, f"stage2_predictions_{split}.csv")
            pred_df.to_csv(pred_path, index=False)
            print(f"[INFO] saved predictions: {pred_path}")

        class_names = ["Class_0", "Class_1"]
        # 打印详细指标
        test_metrics = final_metrics["splits"]["test"]
        print_detailed_metrics(test_metrics, class_names)
        print("\n[TEST] Full metrics:", test_metrics)

        save_json(final_metrics, os.path.join(self.out_dir, "stage2_metrics.json"))
        self._save_summary(final_metrics)
        return final_metrics

    def _save_checkpoint(
            self,
            path: str,
            epoch: int,
            best_score: float,
            metric_for_best: str,
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "cfg": self.cfg,
                "input_dim": self.input_dim,
                "best_score": best_score,
                "metric_for_best": metric_for_best,
            },
            path,
        )


    def _save_summary(self, final_metrics: Dict[str, Any]) -> None:
        path = os.path.join(self.out_dir, "stage2_run_summary.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("Stage2 run summary\n")
            f.write("==================\n")
            f.write(f"best_epoch: {final_metrics['best_epoch']}\n")
            f.write(f"metric_for_best: {final_metrics['metric_for_best']}\n")
            f.write(f"best_score: {final_metrics['best_score']}\n")
            f.write(f"context_method: {self.cfg['context']['method']}\n")
            f.write(f"context_policy: {self.cfg['context']['context_policy']}\n")
            f.write(f"window_size: {self.cfg['context']['window_size']}\n")
            f.write("\nFinal metrics:\n")

            for split in ["train", "val", "test"]:
                metrics = final_metrics["splits"][split]
                f.write(
                    f"{split}: "
                    f"loss={metrics['loss']:.6f}, "
                    f"acc={metrics['accuracy']:.6f}, "
                    f"macro_precision={metrics['macro_precision']:.6f}, "
                    f"macro_recall={metrics['macro_recall']:.6f}, "
                    f"macro_f1={metrics['macro_f1']:.6f}, "
                    f"weighted_precision={metrics['weighted_precision']:.6f}, "
                    f"weighted_recall={metrics['weighted_recall']:.6f}, "
                    f"weighted_f1={metrics['weighted_f1']:.6f}, "
                    f"auc={metrics['auc']}, "
                    f"precision_label1={metrics['precision_label1']}, "
                    f"recall_label1={metrics['recall_label1']}, "
                    f"f1_label1={metrics['f1_label1']}, "
                    f"cm={metrics['confusion_matrix']}\n"
                )

        print(f"[INFO] saved summary: {path}")

def print_detailed_metrics(test_metrics: Dict[str, Any], class_names: List[str] = None) -> None:
    """
    打印详细测试指标，格式与 MLP Baseline 一致。
    """
    print("\n" + "=" * 50)
    print(f"{'Stage2 Transformer - 测试集结果':^50}")
    print("=" * 50)
    print(f"  Loss:              {test_metrics.get('loss', 0):.4f}")
    print(f"  F1 (Macro):        {test_metrics.get('macro_f1', 0):.4f}")
    print(f"  F1 (Weighted):     {test_metrics.get('weighted_f1', 0):.4f}")
    print(f"  AUC (OvR Macro):   {test_metrics.get('auc', 0):.4f}")
    print(f"{'=' * 50}")

    # 各类别 F1
    if "per_class_f1" in test_metrics:
        per_class = test_metrics["per_class_f1"]
        f1_sorted = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
        print("\n[RESULT] 各类别 F1 (降序):")
        for cls, f1 in f1_sorted:
            label_name = class_names[int(cls)] if class_names else cls
            print(f"  {label_name:<30s}: {f1:.4f}")