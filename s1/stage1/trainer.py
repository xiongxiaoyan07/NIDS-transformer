"""
Training and evaluation loop.
"""

from __future__ import annotations

import os
from typing import Dict, Any, List

import torch

from .losses import FocalLoss, compute_class_alpha
from .metrics import classification_metrics
from .utils import save_json


def collect_train_labels(loader) -> List[int]:
    labels = []
    for batch in loader:
        labels.extend(batch["label"].numpy().tolist())
    return labels


def train_model(
    model: torch.nn.Module,
    loaders: Dict[str, Any],
    cfg: Dict[str, Any],
    out_dir: str,
    device: torch.device,
) -> Dict[str, Any]:
    train_cfg = cfg.get("training", {})

    epochs = int(train_cfg.get("epochs", 20))
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    gamma = float(train_cfg.get("focal_gamma", 2.0))
    patience = int(train_cfg.get("early_stop_patience", 5))
    monitor = str(train_cfg.get("monitor_metric", "f1_label1"))

    train_labels = collect_train_labels(loaders["train"])
    alpha = compute_class_alpha(train_labels, num_classes=2).to(device)

    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_score = -1e18
    best_epoch = 0
    epochs_without_improvement = 0

    best_path = os.path.join(out_dir, "stage1_best_model.pt")
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_metrics = evaluate_model(model, loaders["val"], criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        score = float(val_metrics.get(monitor, val_metrics.get("macro_f1", 0.0)))

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_f1_label1={val_metrics['f1_label1']:.4f} "
            f"val_recall_label1={val_metrics['recall_label1']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cfg": cfg,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                },
                best_path,
            )
            print(f"[INFO] saved best model: {best_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[INFO] early stopping at epoch {epoch}")
                break

    save_json({"history": history}, os.path.join(out_dir, "stage1_history.json"))

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate_model(model, loaders["test"], criterion, device)
    save_json(test_metrics, os.path.join(out_dir, "stage1_test_metrics.json"))

    print("[TEST]", test_metrics)

    return {
        "best_model_path": best_path,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "test_metrics": test_metrics,
    }


def _train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        x = batch["x"].to(device)
        t = batch["time"].to(device)
        mask = batch["mask"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x, t, mask)
        loss = criterion(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        total_count += x.size(0)

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_model(model, loader, criterion, device) -> Dict[str, Any]:
    model.eval()

    total_loss = 0.0
    total_count = 0

    y_true = []
    y_pred = []
    y_score = []

    for batch in loader:
        x = batch["x"].to(device)
        t = batch["time"].to(device)
        mask = batch["mask"].to(device)
        y = batch["label"].to(device)

        logits = model(x, t, mask)
        loss = criterion(logits, y)

        prob = torch.softmax(logits, dim=-1)
        pred = torch.argmax(prob, dim=-1)

        total_loss += float(loss.item()) * x.size(0)
        total_count += x.size(0)

        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
        y_score.extend(prob[:, 1].detach().cpu().numpy().tolist())

    metrics = classification_metrics(y_true, y_pred, y_score)
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics
