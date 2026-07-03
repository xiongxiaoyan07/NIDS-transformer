"""
Training and evaluation loop.
"""

from __future__ import annotations

import os
from typing import Dict, Any, List, Tuple

import torch
# ============================================================================
# 在 trainer.py 文件末尾添加以下新函数
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import optuna
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, \
    precision_recall_curve

from torch import nn

warnings.filterwarnings('ignore')

from .losses import FocalLoss, compute_class_alpha, FocalLossWithLabelSmoothing, AsymmetricFocalLoss, \
    ClassBalancedFocalLoss, HardNegativeMiningCELoss
from .metrics import classification_metrics
from .utils import save_json
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR



def collect_train_labels(loader) -> List[int]:
    print("----trainer------collect_train_labels")
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
    train_labels
) -> Dict[str, Any]:
    print("[INFO] trainer.py ------ train_model")

    train_cfg = cfg.get("training", {})

    epochs = int(train_cfg.get("epochs", 20))
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    gamma = float(train_cfg.get("focal_gamma", 2.0))
    patience = int(train_cfg.get("early_stop_patience", 5))
    monitor = str(train_cfg.get("monitor_metric", "f1_label1"))
    label_smoothing = cfg["training"].get("label_smoothing", 0.1)
    use_weighted_sampler = train_cfg.get("use_weighted_sampler", False)

    seq_cfg = cfg.get("sequence", {})
    max_seq_len = int(seq_cfg.get("max_seq_len", 64))
    strategy = seq_cfg.get("strategy", "head")

    alpha_mode = str(train_cfg.get("alpha_mode", "none")).lower()
    loss_name = train_cfg.get("loss", "focal")
    print("[train_model----]loss_name = ",loss_name)
    if alpha_mode == "none":
        alpha = None

    elif alpha_mode == "mild":
        # 温和 class1 加权，不像原始 balanced alpha 那么激进
        alpha = torch.tensor([1, 1.25], dtype=torch.float32, device=device)

    elif alpha_mode == "balanced":
        alpha = compute_class_alpha(train_labels, num_classes=2).to(device)

    else:
        raise ValueError(f"Unknown alpha_mode: {alpha_mode}")

    if loss_name == "ce":
        criterion = nn.CrossEntropyLoss(weight=alpha)
    elif loss_name == "asymmetric":
        criterion = AsymmetricFocalLoss(
            gamma_pos=float(train_cfg.get("asl_gamma_pos", 0.0)),
            gamma_neg=float(train_cfg.get("asl_gamma_neg", 2.0)),
            alpha_pos=float(train_cfg.get("asl_alpha_pos", 0.55)),
        )
    elif loss_name == "cb_focal":
        criterion = ClassBalancedFocalLoss(
            labels=train_labels,
            num_classes=2,
            beta=float(train_cfg.get("cb_beta", 0.999)),
            gamma=float(train_cfg.get("focal_gamma", 1.5)),
            label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
        )
    elif loss_name == "hard_neg_ce":
        criterion = HardNegativeMiningCELoss(
            neg_keep_ratio=float(train_cfg.get("neg_keep_ratio", 0.30)),
            label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
        )
    else:
        criterion = FocalLossWithLabelSmoothing(
            alpha=alpha,
            gamma=gamma,
            label_smoothing=label_smoothing,
        )

    # 类别权重：沿用你原来的不平衡处理
    # alpha = compute_class_alpha(train_labels, num_classes=2).to(device)
    # criterion = FocalLossWithLabelSmoothing(
    #     alpha=alpha,
    #     gamma=gamma,
    #     label_smoothing=label_smoothing,
    # )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # # 添加学习率调度器
    # warmup_epochs = max(1, min(20, epochs // 10))
    # warmup_scheduler = LinearLR(
    #     optimizer,
    #     start_factor=0.1,
    #     end_factor=1.0,
    #     total_iters=warmup_epochs
    # )
    # cosine_scheduler = CosineAnnealingLR(
    #     optimizer,
    #     T_max=epochs - warmup_epochs,
    #     eta_min=lr * 0.01
    # )
    # scheduler = SequentialLR(
    #     optimizer,
    #     schedulers=[warmup_scheduler, cosine_scheduler],
    #     milestones=[warmup_epochs]
    # )
    # 学习率调度
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=epochs
    # )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    best_score = -1e18
    best_epoch = 0
    epochs_without_improvement = 0

    best_path = os.path.join(out_dir, f"seqLen{max_seq_len}{strategy}stage1_best_model.pt")
    history = []

    # 在 train_deep_model 的最开始（optimizer 定义之后）初始化 GradScaler
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if use_weighted_sampler:
        train_loader = loaders["train"]
    else:
        train_loader = loaders["trainNoSampler"]

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        # scheduler.step()  # 添加这行
        val_metrics = evaluate_model(model, loaders["val"], criterion, device)
        scheduler.step(
            val_metrics["pr_auc"]
        )
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        score = float(val_metrics.get(monitor, val_metrics.get("macro_f1", 0.0)))

        if hasattr(model, "flow_fusion") and hasattr(model.flow_fusion, "last_stats"):
            print(f"[Epoch {epoch}] "
                  f"[Fusion stats] -> {model.flow_fusion.last_stats}")
        if hasattr(model, "flow_film") and hasattr(model.flow_film, "last_stats"):
            print(f"[Epoch {epoch}] [TokenFiLM stats] -> {model.flow_film.last_stats}")
        if hasattr(model, "last_flow_token_stats") and model.last_flow_token_stats is not None:
            print(f"[Epoch {epoch}] [FlowToken stats] -> {model.last_flow_token_stats}")
        if (epoch + 1) % 5 == 0:
            print(
                f"[Epoch {epoch + 1}/{epochs}] "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_accuracy={val_metrics['accuracy']:.4f} "
                f"val_f1_label1={val_metrics['f1_label1']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"val_weighted_f1={val_metrics['weighted_f1']:.4f}"
                f"val_auc={val_metrics['auc']:.4f}"
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
            print(f"[INFO] saved best model: {best_path}, best score({monitor}): {best_score}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[INFO] early stopping at epoch {epoch}")
                break

    save_json({"history": history}, os.path.join(out_dir, f"seqLen{max_seq_len}{strategy}stage1_history.json"))

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # 在训练结束后，寻找最优阈值
    print("\n在验证集上寻找最优决策阈值...")
    best_threshold, threshold_results = find_optimal_threshold(
        model, loaders["val"], device, class_idx=1
    )

    # =====================================================================
    # 测试集评估 — 详细输出（与 MLP Baseline 格式一致）
    # =====================================================================
    # test_metrics = evaluate_model(model, loaders["test"], criterion, device)

    # 使用最优阈值重新评估测试集
    print(f"\n使用阈值 {best_threshold:.3f} 评估测试集...")
    test_metrics = evaluate_model(
        model, loaders["test"],
        nn.CrossEntropyLoss(),  # 评估时不需要特殊的损失函数
        device,
        threshold=best_threshold
    )
    # 计算F1上限
    print("[INFO] --- train_model --- 计算F1上限")
    val_labels, val_probs = collect_labels_and_probabilities(
        model,
        loaders["val"],
        device,
    )

    test_labels, test_probs = collect_labels_and_probabilities(
        model,
        loaders["test"],
        device,
    )

    # 1. 验证集当前模型的阈值上限
    val_oracle = find_exact_f1_ceiling(
        val_labels,
        val_probs,
    )

    # 2. 合法评估：使用验证集阈值测试
    test_at_val_threshold = metrics_at_threshold(
        test_labels,
        test_probs,
        val_oracle["threshold"],
    )

    # 3. 测试集阈值上限，只用于诊断
    test_oracle = find_exact_f1_ceiling(
        test_labels,
        test_probs,
    )

    print("\n========== F1 CEILING DIAGNOSIS ==========")

    print("\n[VAL ORACLE]")
    for key, value in val_oracle.items():
        print(f"{key}: {value}")

    print("\n[TEST AT VAL THRESHOLD]")
    for key, value in test_at_val_threshold.items():
        print(f"{key}: {value}")

    print("\n[TEST ORACLE - DIAGNOSTIC ONLY]")
    for key, value in test_oracle.items():
        print(f"{key}: {value}")

    print(
        "\nThreshold transfer gap:",
        test_oracle["f1_label1"]
        - test_at_val_threshold["f1_label1"],
    )
    # 绘制阈值搜索结果
    _plot_threshold_search(threshold_results, out_dir)

    save_json(test_metrics, os.path.join(out_dir, f"seqLen{max_seq_len}{strategy}stage1_test_metrics.json"))

    # 获取类别名称（从数据集中提取）
    try:
        unique_labels = set()
        for batch in loaders["test"]:
            unique_labels.update(batch["label"].numpy().tolist())
        class_names = [f"Class_{i}" for i in sorted(unique_labels)]
    except:
        class_names = ["Class_0", "Class_1"]

    # 打印详细指标
    print_detailed_metrics(test_metrics, class_names)

    # =====================================================================
    # 可视化
    # =====================================================================
    # 训练曲线
    plot_training_curves(
        history=history,
        out_dir=out_dir,
        best_epoch=best_epoch,
        prefix=f"seqLen{max_seq_len}{strategy}stage1"
    )

    # 混淆矩阵
    if "confusion_matrix" in test_metrics:
        plot_confusion_matrix(
            cm=test_metrics["confusion_matrix"],
            class_names=class_names,
            out_dir=out_dir,
            prefix=f"seqLen{max_seq_len}{strategy}stage1"
        )

    # 各类别 F1
    if "per_class_f1" in test_metrics:
        plot_per_class_f1(
            per_class_f1=test_metrics["per_class_f1"],
            class_names=class_names,
            out_dir=out_dir,
            prefix=f"seqLen{max_seq_len}{strategy}stage1"
        )

    print("\n[TEST] Full metrics:", test_metrics)

    return {
        "best_model_path": best_path,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "test_metrics": test_metrics,
    }

def _train_one_epoch(model, loader, optimizer, criterion, device, scaler) -> float:
    model.train()

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        t = batch["time"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        # 获取flow_feats（如果存在）
        flow_feats = batch.get("flow_feats")
        if flow_feats is not None:
            flow_feats = flow_feats.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # 调用模型时传入flow_feats
        # 🚀 使用混合精度向前传播
        use_amp = device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x, t, mask, flow_feats=flow_feats)
            loss = criterion(logits, y)

        # 🚀 混合精度反向传播
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())

    return total_loss / len(loader)

def train_model_for_optuna(
    model: torch.nn.Module,
    loaders: Dict[str, Any],
    cfg: Dict[str, Any],
    out_dir: str,
    device: torch.device,
    train_labels,
    trial=None,
    save_best: bool = False,
) -> Dict[str, Any]:
    """
    Optuna 专用训练函数。

    与 train_model() 的区别：
    1. 只使用 train / val。
    2. 不评估 test，避免 HPO 阶段污染 test set。
    3. 不做 threshold_search，不画图，减少 trial 开销。
    4. 支持 Optuna pruning。
    5. 返回 best_score / best_epoch / history / best_val_metrics。
    """
    print("\n[TRAIN] Training model... train_model_for_optuna")
    os.makedirs(out_dir, exist_ok=True)

    train_cfg = cfg.get("training", {})
    seq_cfg = cfg.get("sequence", {})

    epochs = int(train_cfg.get("epochs", 80))
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    gamma = float(train_cfg.get("focal_gamma", 2.0))
    patience = int(train_cfg.get("early_stop_patience", 10))
    monitor = str(train_cfg.get("monitor_metric", "f1_label1"))
    monitor_mode = str(train_cfg.get("monitor_mode", "max")).lower()
    label_smoothing = float(train_cfg.get("label_smoothing", 0.05))

    max_seq_len = int(seq_cfg.get("max_seq_len", 64))
    strategy = seq_cfg.get("strategy", "head")

    alpha_mode = str(train_cfg.get("alpha_mode", "none")).lower()
    loss_name = train_cfg.get("loss", "focal")

    if alpha_mode == "none":
        alpha = None

    elif alpha_mode == "mild":
        # 温和 class1 加权，不像原始 balanced alpha 那么激进
        alpha = torch.tensor([0.4, 0.6], dtype=torch.float32, device=device)

    elif alpha_mode == "balanced":
        alpha = compute_class_alpha(train_labels, num_classes=2).to(device)

    else:
        raise ValueError(f"Unknown alpha_mode: {alpha_mode}")
    # 类别权重：沿用你原来的不平衡处理
    # alpha = compute_class_alpha(train_labels, num_classes=2).to(device)
    if loss_name == "ce":
        criterion = nn.CrossEntropyLoss(weight=alpha)
    else:
        criterion = FocalLossWithLabelSmoothing(
            alpha=alpha,
            gamma=gamma,
            label_smoothing=label_smoothing,
        )
    # criterion = FocalLossWithLabelSmoothing(
    #     alpha=alpha,
    #     gamma=gamma,
    #     label_smoothing=label_smoothing,
    # )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # 你的原代码 warmup_epochs = min(10, epochs // 10)。
    # 这里加 max(1, ...) 防止 epochs 较小时 warmup_epochs=0。
    warmup_epochs = max(1, min(10, epochs // 10))
    cosine_epochs = max(1, epochs - warmup_epochs)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=lr * 0.01,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    if monitor_mode == "min":
        best_score = float("inf")
        is_better = lambda score, best: score < best
    else:
        best_score = -float("inf")
        is_better = lambda score, best: score > best

    best_epoch = 0
    best_val_metrics = None
    epochs_without_improvement = 0
    history = []

    best_path = os.path.join(
        out_dir,
        f"trial_best_seqLen{max_seq_len}_{strategy}.pt",
    )

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        val_metrics = evaluate_model(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
        )

        score = float(val_metrics.get(monitor, val_metrics.get("macro_f1", 0.0)))

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "monitor": monitor,
            "score": score,
        }
        history.append(row)

        print(
            f"[Optuna Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_weighted_f1={val_metrics['weighted_f1']:.4f} "
            f"val_f1_label1={val_metrics.get('f1_label1', 0.0):.4f} "
            f"val_auc={val_metrics.get('auc', 0.0):.4f}"
        )

        # Optuna pruning：把每个 epoch 的验证分数汇报给 trial
        if trial is not None:
            trial.report(score, step=epoch)

            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"Pruned at epoch={epoch}, {monitor}={score:.6f}"
                )

        if is_better(score, best_score):
            best_score = score
            best_epoch = epoch
            best_val_metrics = val_metrics
            epochs_without_improvement = 0

            if save_best:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "cfg": cfg,
                        "best_epoch": best_epoch,
                        "best_score": best_score,
                        "best_val_metrics": best_val_metrics,
                    },
                    best_path,
                )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[Optuna] early stopping at epoch {epoch}")
                break

    result = {
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "best_val_metrics": best_val_metrics,
        "history": history,
    }

    if save_best:
        result["best_model_path"] = best_path

    save_json(result, os.path.join(out_dir, "optuna_trial_summary.json"))

    return result

def _plot_threshold_search(results, out_dir):
    """绘制阈值搜索曲线"""
    import matplotlib.pyplot as plt

    thresholds = [r['threshold'] for r in results]
    f1_scores = [r['f1_macro'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, 'b-', label='F1 Score', linewidth=2)
    plt.plot(thresholds, precisions, 'g--', label='Precision')
    plt.plot(thresholds, recalls, 'r--', label='Recall')

    plt.xlabel('Decision Threshold')
    plt.ylabel('Score')
    plt.title('Threshold Optimization for Attack Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 标记最优阈值
    best_idx = f1_scores.index(max(f1_scores))
    plt.axvline(x=thresholds[best_idx], color='k', linestyle=':', alpha=0.5)
    plt.text(thresholds[best_idx], max(f1_scores),
             f'Best: {thresholds[best_idx]:.2f}',
             ha='center', va='bottom')

    plt.savefig(f"{out_dir}/threshold_optimization.png", dpi=150, bbox_inches='tight')
    plt.close()

@torch.no_grad()
def evaluate_model(
        model: nn.Module,
        loader,
        criterion: nn.Module,
        device: torch.device,
        threshold: float = None,  # 新增参数
) -> Dict[str, Any]:
    """
    评估模型，支持自定义决策阈值
    """
    model.eval()
    y_true = []
    y_pred = []
    y_score = []
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        t = batch["time"].to(device)
        mask = batch["mask"].to(device)
        y = batch["label"].to(device)

        # 获取flow_feats（如果存在）
        flow_feats = batch.get("flow_feats")
        if flow_feats is not None:
            flow_feats = flow_feats.to(device)

        with torch.no_grad():
            logits = model(x, t, mask, flow_feats=flow_feats)
            loss = criterion(logits, y)

            probs = torch.softmax(logits, dim=-1)

            # 如果提供了阈值，使用自定义阈值
            if threshold is not None:
                preds = (probs[:, 1] >= threshold).long()
            else:
                preds = logits.argmax(dim=-1)

        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_score.extend(probs[:, 1].cpu().numpy().tolist())
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches

    # 计算详细指标
    metrics = classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        num_classes=2,
        loss=avg_loss,
        threshold=threshold,
    )

    return metrics


def find_optimal_threshold(model,
                           val_loader,
                           device,
                           class_idx=1,
                           target_precision=0.85,
                           target_recall=0.85,
                       ):
    """
    在验证集上寻找最优决策阈值
    """
    model.eval()
    y_true = []
    y_scores = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            t = batch["time"].to(device)
            mask = batch["mask"].to(device)
            y = batch["label"].to(device)

            # 获取flow_feats（如果存在）
            flow_feats = batch.get("flow_feats")
            if flow_feats is not None:
                flow_feats = flow_feats.to(device)

            logits = model(x, t, mask, flow_feats=flow_feats)
            probs = torch.softmax(logits, dim=-1)

            y_true.extend(y.cpu().numpy().tolist())
            y_scores.extend(probs[:, class_idx].cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # 网格搜索最优阈值
    thresholds = np.arange(0.1, 0.95, 0.001)
    results = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_label1 = f1_score(y_true, y_pred, pos_label=1)
        precision_label1 =  precision_score(y_true,y_pred,pos_label=1,zero_division=0)
        recall_label1 = recall_score(y_true,y_pred,pos_label=1,zero_division=0)

        results.append({
            'threshold': threshold,
            'f1': f1_label1,
            "precision_label1": precision_label1,
            "recall_label1": recall_label1,
            "f1_macro": f1,
            'precision': precision,
            'recall': recall,
        })

    feasible = [
        row for row in results
        if row["precision_label1"] >= target_precision
           and row["recall_label1"] >= target_recall
    ]

    if feasible:
        best = max(
            feasible,
            key=lambda row: row["f1"],
        )
        print("找到满足 P>=0.85 且 R>=0.85 的验证阈值")
    else:
        best = max(
            results,
            key=lambda row: min(
                row["precision_label1"],
                row["recall_label1"],
            ),
        )
        print("验证集上不存在同时满足 P>=0.85、R>=0.85 的阈值")
        print("当前最平衡的结果为：")

    print(best)

    # 选择F1最高的阈值
    best_result = max(results, key=lambda x: x['f1'])

    print(f"\n[THRESHOLD] 最优阈值: {best_result['threshold']:.2f}")
    print(f"  F1_class1: {best_result['f1']:.4f}")
    print(f"  F1_macro: {best_result['f1_macro']:.4f}")
    print(f"  Precision_macro: {best_result['precision']:.4f}")
    print(f"  Recall_macro: {best_result['recall']:.4f}")

    return best_result['threshold'], results

def print_detailed_metrics(test_metrics: Dict[str, Any], class_names: List[str] = None) -> None:
    """
    打印详细测试指标，格式与 MLP Baseline 一致。
    """
    print("\n" + "=" * 50)
    print(f"{'Stage1 Transformer - 测试集结果':^50}")
    print("=" * 50)
    print(f"  Loss:              {test_metrics.get('loss', 0):.4f}")
    print(f"  F1_label1:         {test_metrics.get('f1_label1', 0):.4f}")
    print(f"  Precision_label1:  {test_metrics.get('precision_label1', 0):.4f}")
    print(f"  Recall_label1:     {test_metrics.get('recall_label1', 0):.4f}")
    print(f"  F1 (Macro):        {test_metrics.get('macro_f1', 0):.4f}")
    print(f"  Recall (Macro):    {test_metrics.get('macro_recall', 0):.4f}")
    print(f"  Precision (Macro): {test_metrics.get('macro_precision', 0):.4f}")
    print(f"  F1 (Weighted):     {test_metrics.get('weighted_f1', 0):.4f}")
    print(f"  AUC (OvR Macro):   {test_metrics.get('auc', 0):.4f}")
    print(f"  PR_AUC:            {test_metrics.get('pr_auc', 0):.4f}")
    print(f"{'=' * 50}")

    # 各类别 F1
    if "per_class_f1" in test_metrics:
        per_class = test_metrics["per_class_f1"]
        f1_sorted = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
        print("\n[RESULT] 测试集 --- 各类别 F1 (降序):")
        for cls, f1 in f1_sorted:
            label_name = class_names[int(cls)] if class_names else cls
            print(f"  {label_name:<30s}: {f1:.4f}")


def plot_training_curves(history: List[Dict], out_dir: str, best_epoch: int,
                         prefix: str = "stage1") -> None:
    """
    绘制训练曲线：Loss 和 F1。
    """
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h.get("val_loss", 0) for h in history]
    val_f1 = [h.get("val_macro_f1", 0) for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1: Loss 曲线
    ax1 = axes[0]
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    if best_epoch > 0:
        ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7,
                    label=f'Best Epoch ({best_epoch})')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 子图2: F1 曲线
    ax2 = axes[1]
    ax2.plot(epochs, val_f1, 'g-', label='Val F1 (Macro)', linewidth=2)
    if best_epoch > 0:
        ax2.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7,
                    label=f'Best Epoch ({best_epoch})')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('F1 Score (Macro)', fontsize=12)
    ax2.set_title('Validation F1 Score', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{prefix}_training_curves.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"[INFO] 训练曲线已保存: {save_path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_dir: str,
                          prefix: str = "stage1") -> None:
    """
    绘制归一化混淆矩阵。
    """
    cm = np.array(cm)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{prefix}_confusion_matrix.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"[INFO] 混淆矩阵已保存: {save_path}")


def plot_per_class_f1(per_class_f1: Dict[str, float], class_names: List[str],
                      out_dir: str, prefix: str = "stage1") -> None:
    """
    绘制各类别 F1 分数条形图。
    """
    # 按 F1 排序
    items = sorted(per_class_f1.items(), key=lambda x: x[1])
    names = [class_names[int(k)] if class_names else k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(values)))
    bars = ax.barh(range(len(values)), values, color=colors, edgecolor='gray', linewidth=0.5)

    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('F1 Score', fontsize=12)
    ax.set_title('Per-Class F1 Score (Sorted)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='F1=0.5')

    for bar, val in zip(bars, values):
        ax.text(min(val + 0.02, 1.02), bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{prefix}_per_class_f1.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"[INFO] 各类别 F1 图已保存: {save_path}")


@torch.inference_mode()
def collect_labels_and_probabilities(
        model: torch.nn.Module,
        loader,
        device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """收集真实标签和 class-1 概率。"""
    model.eval()

    all_labels = []
    all_probs = []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        time_log = batch["time"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        flow_feats = batch.get("flow_feats")
        if flow_feats is not None:
            flow_feats = flow_feats.to(device, non_blocking=True)

        logits = model(
            x,
            time_log,
            mask,
            flow_feats=flow_feats,
        )

        probs_class1 = torch.softmax(logits, dim=-1)[:, 1]

        all_labels.append(batch["label"].detach().cpu().numpy())
        all_probs.append(probs_class1.detach().cpu().numpy())

    labels = np.concatenate(all_labels).astype(np.int64)
    probabilities = np.concatenate(all_probs).astype(np.float64)

    return labels, probabilities

def metrics_at_threshold(
        labels: np.ndarray,
        probabilities: np.ndarray,
        threshold: float,
) -> Dict[str, Any]:
    predictions = (probabilities >= threshold).astype(np.int64)

    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (
            2.0 * precision * recall / max(precision + recall, 1e-12)
    )

    return {
        "threshold": float(threshold),
        "precision_label1": float(precision),
        "recall_label1": float(recall),
        "f1_label1": float(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }

def find_exact_f1_ceiling(
        labels: np.ndarray,
        probabilities: np.ndarray,
) -> Dict[str, Any]:
    """
    找到当前概率排序下，通过单一阈值能够获得的最高 class-1 F1。

    这是 threshold oracle，不代表模型可部署性能。
    """
    precision, recall, thresholds = precision_recall_curve(
        labels,
        probabilities,
    )

    # precision/recal 比 thresholds 多一个元素。
    precision_t = precision[:-1]
    recall_t = recall[:-1]

    f1 = (
            2.0
            * precision_t
            * recall_t
            / np.maximum(precision_t + recall_t, 1e-12)
    )

    best_index = int(np.nanargmax(f1))
    best_threshold = float(thresholds[best_index])

    result = metrics_at_threshold(
        labels,
        probabilities,
        best_threshold,
    )

    result.update({
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(
            average_precision_score(labels, probabilities)
        ),
        "num_samples": int(len(labels)),
        "num_positive": int(np.sum(labels == 1)),
        "num_negative": int(np.sum(labels == 0)),
    })

    return result