from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import Stage2Dataset, stage2_collate_fn
from .losses import compute_class_alpha,FocalLossWithLabelSmoothing
from .utils import metric_value, save_json, worker_init_fn
from .metrics import classification_metrics
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from functools import partial


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def model_parameter_profile(model: nn.Module) -> Dict[str, Any]:
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    non_trainable = total - trainable
    return {
        "params_total": total,
        "params_trainable": trainable,
        "params_non_trainable": non_trainable,
        "estimated_fp32_size_mb": float(total * 4 / (1024 ** 2)),
    }

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
            print(f"[INFO]manual_alpha = {manual_alpha}")
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

def make_stage2_collate(cfg):
    fixed_max_len = max(1, int(cfg["context"].get("window_size", 16)))
    return partial(stage2_collate_fn, fixed_max_len=fixed_max_len)

def make_train_loader(dataset: Stage2Dataset, cfg: Dict[str, Any]) -> DataLoader:
    train_cfg = cfg["training"]
    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 0))
    collate = make_stage2_collate(cfg)
    # 不要同时使用 WeightedRandomSampler 和类别加权 Focal Loss 当前配置:use_weighted_sampler = False
    if bool(train_cfg.get("use_weighted_sampler", True)):
        labels_np = dataset.labels.astype(int)

        n0 = int((labels_np == 0).sum())
        n1 = int((labels_np == 1).sum())

        target_pos_frac = float(train_cfg.get("sampler_pos_fraction", 0.20))

        w0 = (1.0 - target_pos_frac) / max(n0, 1)
        w1 = target_pos_frac / max(n1, 1)

        sample_weights = np.where(labels_np == 1, w1, w0).astype(np.float64)

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(int(cfg.get("seed", 42))), # 固定种子
        )

        print(
            "[INFO] make_train_loader: "
            f"n0={n0}, n1={n1}, target_pos_frac={target_pos_frac:.3f}, "
            f"w0={w0:.6g}, w1={w1:.6g}, sample_weights={sample_weights}"
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate,
            worker_init_fn=worker_init_fn,
            pin_memory=torch.cuda.is_available(),
            generator=torch.Generator().manual_seed(int(cfg.get("seed", 42))),
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(int(cfg.get("seed", 42))),
    )

def make_eval_loader(dataset: Stage2Dataset, cfg: Dict[str, Any]) -> DataLoader:
    train_cfg = cfg["training"]
    collate = make_stage2_collate(cfg)
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        worker_init_fn=worker_init_fn,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(int(cfg.get("seed", 42))),
    )

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip_norm: Optional[float],
    scaler: Optional[torch.amp.GradScaler],
) -> float:
    model.train()
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_n = 0

    for batch in loader:
        x = batch["context_z"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        amp_enabled = scaler is not None and scaler.is_enabled()
        amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model(x, mask)
            loss = loss_fn(logits, y)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()

            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

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
    threshold: Optional[float] = 0.5,
    return_predictions: bool = True,
    amp_enabled=False, amp_dtype=torch.float16
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    model.eval()

    losses: List[float] = []
    all_y: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_score: List[np.ndarray] = []
    all_flow_ids: List[int] = []
    for batch in loader:
        x = batch["context_z"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
        ):
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
        all_y.append(y.detach())
        all_pred.append(preds.detach())
        all_score.append(score_label1.detach())
        all_flow_ids.extend(batch["flow_id"].cpu().numpy().tolist())

    y_true = torch.cat(all_y).cpu().numpy()
    y_pred = torch.cat(all_pred).cpu().numpy()
    y_score = torch.cat(all_score).cpu().numpy()
    avg_loss = float(np.sum(losses) / max(len(y_true), 1))

    metrics = classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        num_classes=2,
        loss=avg_loss,
        threshold=threshold,
    )

    metrics["loss"] = avg_loss
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
    if return_predictions:
        return metrics, pred_df
    else:
        return metrics

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

        if metric in {"f1", "f1_label1"}:
            value = metrics["f1_label1"]
        elif metric == "recall":
            value = metrics["recall_label1"]
        elif metric == "precision":
            value = metrics["precision_label1"]
        elif metric == "macro_f1":
            value = metrics["macro_f1"]
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
        self.model_profile = model_parameter_profile(self.model)
        self.training_profile: Dict[str, Any] = {}
        self.inference_profile: Dict[str, Any] = {}

        print(
            "[INFO] Stage2 model profile: "
            f"params_total={self.model_profile['params_total']:,}, "
            f"params_trainable={self.model_profile['params_trainable']:,}, "
            f"estimated_fp32_size_mb={self.model_profile['estimated_fp32_size_mb']:.3f}"
        )

        self.train_loader = make_train_loader(datasets["train"], cfg)
        self.eval_loaders = {
            "train": make_eval_loader(datasets["train"], cfg),
            "val": make_eval_loader(datasets["val"], cfg),
            "test": make_eval_loader(datasets["test"], cfg),
        }

        self.loss_fn = make_loss_fn(datasets["train"].labels, cfg, device)

        optimizer_kwargs = {
            "lr": float(cfg["training"].get("lr", 3e-4)),
            "weight_decay": float(cfg["training"].get("weight_decay", 1e-4)),
        }

        if device.type == "cuda":
            optimizer_kwargs["fused"] = True

        try:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                **optimizer_kwargs,
            )
        except (TypeError, RuntimeError):
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(cfg["training"].get("lr", 3e-4)),
                weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
            )
    # train model
    def fit(self) -> Dict[str, Any]:
        _sync_device(self.device)
        epochs = int(self.cfg["training"].get("epochs", 100))
        patience = int(self.cfg["training"].get("patience", 20))
        threshold = float(self.cfg["training"].get("threshold", 0.5))
        metric_for_best = self.cfg["training"].get("metric_for_best", "f1_label1")

        grad_clip_norm = self.cfg["training"].get("grad_clip_norm", 1.0)
        grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)

        lr = float(self.cfg["training"].get("lr", 3e-4))
        weight_decay = float(self.cfg["training"].get("weight_decay", 3e-4))
        print(f"[Stage2Trainer]---fit lr={lr}, weight_decay={weight_decay}, patience={patience}, threshold={threshold}, metric_for_best={metric_for_best}")
        # 你的原代码 warmup_epochs = min(10, epochs // 10)。
        # 这里加 max(1, ...) 防止 epochs 较小时 warmup_epochs=0。
        warmup_epochs = max(1, min(5, epochs // 10))

        cosine_epochs = max(1, epochs - warmup_epochs)
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.05,
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

        amp_enabled = self.device.type == "cuda"
        scaler = (
            torch.amp.GradScaler("cuda", enabled=amp_enabled)
            if self.device.type == "cuda"
            else None
        )
        best_score = -float("inf")
        best_epoch = -1
        bad_epochs = 0
        epoch_rows: List[Dict[str, Any]] = []
        last_epoch = 0

        fit_start = time.perf_counter()
        for epoch in range(1, epochs + 1):
            last_epoch = epoch
            train_loss = train_one_epoch(
                model=self.model,
                loader=self.train_loader,
                optimizer=self.optimizer,
                loss_fn=self.loss_fn,
                device=self.device,
                grad_clip_norm=grad_clip_norm,
                scaler=scaler
            )

            metrics_by_split = {}

            metrics_05, val_pred_df = evaluate(
                model=self.model,
                loader=self.eval_loaders["val"],
                loss_fn=self.loss_fn,
                device=self.device,
                threshold=0.5,
                return_predictions=True,
                amp_enabled=(self.device.type == "cuda"),
            )

            if bool(self.cfg["training"].get("select_threshold_each_epoch", True)):
                # print("********select_threshold_each_epoch*********")
                best_th, best_th_metrics = find_best_threshold(
                    y_true=val_pred_df["label"].to_numpy(),
                    y_score=val_pred_df["prob_label_1"].to_numpy(),
                    metric=self.cfg["training"].get("threshold_metric", "f1_label1"),
                    threshold_min=float(self.cfg["training"].get("threshold_min", 0.01)),
                    threshold_max=float(self.cfg["training"].get("threshold_max", 0.99)),
                    threshold_steps=int(self.cfg["training"].get("threshold_steps", 199)),
                )
                print("fit------best threshold = ", best_th)
                metrics = dict(best_th_metrics)
                metrics["selected_threshold"] = float(best_th)

                # 保留 ranking metrics
                metrics["loss"] = metrics_05["loss"]
                metrics["auc"] = metrics_05["auc"]
                metrics["pr_auc"] = metrics_05["pr_auc"]
                metrics["num_samples"] = metrics_05["num_samples"]
                metrics["positive_count"] = metrics_05["positive_count"]
                metrics["positive_rate"] = metrics_05["positive_rate"]
            else:
                metrics = metrics_05

            metrics_by_split["val"] = metrics
            score = metric_value(metrics_by_split, metric_for_best)
            scheduler.step()
            min_delta = float(self.cfg["training"].get("min_delta", 0.0))
            improved = score > best_score + min_delta
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

            current_lr = float(self.optimizer.param_groups[0]["lr"])
            row = {
                "epoch": epoch,
                "train_loss_optim": train_loss,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "current_lr": current_lr,
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
                f"val_f1_label1={metrics_by_split['val']['f1_label1']:.4f} "
                f"val_macro_f1={metrics_by_split['val']['macro_f1']:.4f} "
                f"val_weighted_f1={metrics_by_split['val']['weighted_f1']} "
                f"val_pr_auc={metrics_by_split['val']['pr_auc']:.4f} "
                f"val_auc={metrics_by_split['val']['auc']:.4f}"
                f"lr={current_lr:.8f} "
                f"{'*' if improved else ''}"
            )

            if 0 < patience <= bad_epochs:
                print(f"[INFO] Early stopping at epoch={epoch}, best_epoch={best_epoch}")
                break

        fit_seconds = time.perf_counter() - fit_start
        _sync_device(self.device)
        epochs_completed = int(last_epoch)
        self.training_profile = {
            "training_time_seconds": float(fit_seconds),
            "epochs_completed": epochs_completed,
            "best_epoch": int(best_epoch),
            "best_score": float(best_score),
            "early_stopped": bool(0 < patience <= bad_epochs),
            "avg_seconds_per_epoch": float(fit_seconds / max(epochs_completed, 1)),
        }
        print(
            "[INFO] Stage2 training profile: "
            f"training_time_seconds={fit_seconds:.2f}, "
            f"epochs_completed={epochs_completed}, "
            f"avg_seconds_per_epoch={self.training_profile['avg_seconds_per_epoch']:.2f}"
        )
        return self.training_profile

    def threshold_oracle_diagnosis(
            self,
            val_threshold: float,
    ) -> Dict[str, Any]:
        """
        Diagnostic only.

        Purpose:
        1. Use validation-selected threshold on test set.
        2. Also compute test oracle threshold.
        3. Measure threshold transfer gap.

        This should NOT be used as the official test result.
        The official result must use val_threshold.
        """

        _, val_pred_df = evaluate(
            model=self.model,
            loader=self.eval_loaders["val"],
            loss_fn=self.loss_fn,
            device=self.device,
            threshold=0.5,
            return_predictions=True,
            amp_enabled=(self.device.type == "cuda"),
        )

        _, test_pred_df = evaluate(
            model=self.model,
            loader=self.eval_loaders["test"],
            loss_fn=self.loss_fn,
            device=self.device,
            threshold=0.5,
            return_predictions=True,
            amp_enabled=(self.device.type == "cuda"),
        )

        y_val = val_pred_df["label"].to_numpy()
        p_val = val_pred_df["prob_label_1"].to_numpy()

        y_test = test_pred_df["label"].to_numpy()
        p_test = test_pred_df["prob_label_1"].to_numpy()

        val_oracle_threshold, val_oracle_metrics = find_best_threshold(
            y_true=y_val,
            y_score=p_val,
            metric=self.cfg["training"].get("threshold_metric", "f1_label1"),
            threshold_min=float(self.cfg["training"].get("threshold_min", 0.01)),
            threshold_max=float(self.cfg["training"].get("threshold_max", 0.99)),
            threshold_steps=int(self.cfg["training"].get("threshold_steps", 199)),
        )

        test_at_val_metrics = classification_metrics(
            y_true=y_test,
            y_pred=(p_test >= float(val_threshold)).astype(int),
            y_score=p_test,
            num_classes=2,
            threshold=float(val_threshold),
        )

        test_oracle_threshold, test_oracle_metrics = find_best_threshold(
            y_true=y_test,
            y_score=p_test,
            metric=self.cfg["training"].get("threshold_metric", "f1_label1"),
            threshold_min=float(self.cfg["training"].get("threshold_min", 0.01)),
            threshold_max=float(self.cfg["training"].get("threshold_max", 0.99)),
            threshold_steps=int(self.cfg["training"].get("threshold_steps", 199)),
        )

        threshold_transfer_gap = (
                float(test_oracle_metrics["f1_label1"])
                - float(test_at_val_metrics["f1_label1"])
        )

        diagnosis = {
            "val_oracle_threshold": float(val_oracle_threshold),
            "val_oracle_metrics": val_oracle_metrics,
            "val_selected_threshold": float(val_threshold),
            "test_at_val_threshold": test_at_val_metrics,
            "test_oracle_threshold": float(test_oracle_threshold),
            "test_oracle_metrics": test_oracle_metrics,
            "threshold_transfer_gap": threshold_transfer_gap,
        }

        print("\n========== STAGE2 THRESHOLD ORACLE DIAGNOSIS ==========")
        print(f"[VAL ORACLE THRESHOLD] {val_oracle_threshold:.6f}")
        print("[VAL ORACLE METRICS]", val_oracle_metrics)

        print(f"\n[TEST AT VAL THRESHOLD] {val_threshold:.6f}")
        print("[TEST AT VAL THRESHOLD METRICS]", test_at_val_metrics)

        print(f"\n[TEST ORACLE THRESHOLD - DIAGNOSTIC ONLY] {test_oracle_threshold:.6f}")
        print("[TEST ORACLE METRICS]", test_oracle_metrics)

        print(f"\n[THRESHOLD TRANSFER GAP] {threshold_transfer_gap:.6f}")
        print("=======================================================\n")

        save_json(
            diagnosis,
            os.path.join(self.out_dir, "stage2_threshold_oracle_diagnosis.json"),
        )

        return diagnosis
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
                metric=self.cfg["training"].get("threshold_metric", "f1_label1"),
                threshold_min=float(self.cfg["training"].get("threshold_min", 0.01)),
                threshold_max=float(self.cfg["training"].get("threshold_max", 0.99)),
                threshold_steps=int(self.cfg["training"].get("threshold_steps", 199)),
            )

            self.cfg["training"]["threshold"] = threshold
            print(f"[INFO] Auto-selected threshold on val: {threshold:.4f}")
            print(f"[INFO] Val metrics at selected threshold: {threshold_metrics}")
        # ============================================================
        # Diagnostic only: threshold transfer / test oracle
        # Must be after load_best() and after val threshold selection.
        # ============================================================
        threshold_diagnosis = self.threshold_oracle_diagnosis(
            val_threshold=threshold,
        )
        context_lengths = np.array([len(x) for x in context_indices], dtype=np.int64)
        final_metrics: Dict[str, Any] = {
            "best_epoch": int(ckpt["epoch"]),
            "best_score": float(ckpt["best_score"]),
            "metric_for_best": ckpt["metric_for_best"],
            "threshold": float(threshold),
            "threshold_diagnosis": threshold_diagnosis,
            "model_profile": self.model_profile,
            "training_profile": self.training_profile,
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
            "inference_profile": {"splits": {}},
        }

        for split, loader in self.eval_loaders.items():
            _sync_device(self.device)
            infer_start = time.perf_counter()
            metrics, pred_df = evaluate(
                model=self.model,
                loader=loader,
                loss_fn=self.loss_fn,
                device=self.device,
                threshold=threshold,
            )
            _sync_device(self.device)
            infer_seconds = time.perf_counter() - infer_start
            final_metrics["splits"][split] = metrics
            num_samples = int(metrics.get("num_samples", 0))
            final_metrics["inference_profile"]["splits"][split] = {
                "inference_time_seconds": float(infer_seconds),
                "num_samples": num_samples,
                "num_batches": int(len(loader)),
                "seconds_per_sample": float(infer_seconds / max(num_samples, 1)),
                "samples_per_second": float(num_samples / max(infer_seconds, 1e-12)),
            }

            pred_path = os.path.join(self.out_dir, f"stage2_predictions_{split}.csv")
            pred_df.to_csv(pred_path, index=False)
            print(f"[INFO] saved predictions: {pred_path}")

        test_infer = final_metrics["inference_profile"]["splits"].get("test", {})
        final_metrics["efficiency"] = {
            "params_total": int(self.model_profile["params_total"]),
            "params_trainable": int(self.model_profile["params_trainable"]),
            "params_non_trainable": int(self.model_profile["params_non_trainable"]),
            "estimated_fp32_size_mb": float(self.model_profile["estimated_fp32_size_mb"]),
            "training_time_seconds": self.training_profile.get("training_time_seconds"),
            "epochs_completed": self.training_profile.get("epochs_completed"),
            "avg_seconds_per_epoch": self.training_profile.get("avg_seconds_per_epoch"),
            "inference_time_seconds_test": test_infer.get("inference_time_seconds"),
            "test_seconds_per_sample": test_infer.get("seconds_per_sample"),
            "test_samples_per_second": test_infer.get("samples_per_second"),
        }

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
            efficiency = final_metrics.get("efficiency", {})
            inference_splits = final_metrics.get("inference_profile", {}).get("splits", {})
            f.write("\nEfficiency:\n")
            f.write(f"params_total: {efficiency.get('params_total')}\n")
            f.write(f"params_trainable: {efficiency.get('params_trainable')}\n")
            f.write(f"params_non_trainable: {efficiency.get('params_non_trainable')}\n")
            f.write(f"estimated_fp32_size_mb: {efficiency.get('estimated_fp32_size_mb')}\n")
            f.write(f"training_time_seconds: {efficiency.get('training_time_seconds')}\n")
            f.write(f"epochs_completed: {efficiency.get('epochs_completed')}\n")
            f.write(f"avg_seconds_per_epoch: {efficiency.get('avg_seconds_per_epoch')}\n")
            f.write(f"inference_time_seconds_test: {efficiency.get('inference_time_seconds_test')}\n")
            f.write(f"test_seconds_per_sample: {efficiency.get('test_seconds_per_sample')}\n")
            f.write(f"test_samples_per_second: {efficiency.get('test_samples_per_second')}\n")
            for split in ["train", "val", "test"]:
                split_profile = inference_splits.get(split, {})
                f.write(
                    f"inference_{split}: "
                    f"seconds={split_profile.get('inference_time_seconds')}, "
                    f"samples={split_profile.get('num_samples')}, "
                    f"seconds_per_sample={split_profile.get('seconds_per_sample')}, "
                    f"samples_per_second={split_profile.get('samples_per_second')}\n"
                )
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
                    f"pr_auc={metrics['pr_auc']}, "
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
    print(f"  Recall (Macro):    {test_metrics.get('macro_recall', 0):.4f}")
    print(f"  Precision (Macro): {test_metrics.get('macro_precision', 0):.4f}")
    print(f"  F1_class1:         {test_metrics.get('f1_label1', 0):.4f}")
    print(f"  R1_class1:         {test_metrics.get('recall_label1', 0):.4f}")
    print(f"  P1_class1:         {test_metrics.get('precision_label1', 0):.4f}")
    print(f"  AUC (OvR Macro):   {test_metrics.get('auc', 0):.4f}")
    print(f"  PR_AUC:            {test_metrics.get('pr_auc', 0):.4f}")
    print(f"{'=' * 50}")

    # 各类别 F1
    if "per_class_f1" in test_metrics:
        per_class = test_metrics["per_class_f1"]
        f1_sorted = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
        print("\n[RESULT] 各类别 F1 (降序):")
        for cls, f1 in f1_sorted:
            label_name = class_names[int(cls)] if class_names else cls
            print(f"  {label_name:<30s}: {f1:.4f}")
