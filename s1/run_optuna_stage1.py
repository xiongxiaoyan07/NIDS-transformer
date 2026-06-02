#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import gc
import os
from typing import Any, Dict

import numpy as np
import optuna
import torch
import yaml

from torch.utils.data import DataLoader, WeightedRandomSampler

from stage1.config import load_config
from stage1.model import Stage1TimeAwareTransformer
from stage1.pipeline import build_dataloaders, custom_collate_fn
from stage1.trainer import train_model_for_optuna
from stage1.utils import set_seed, safe_mkdir, save_json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--packet_csv", required=True)
    parser.add_argument("--flow_csv", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", default="./stage1_optuna")

    parser.add_argument("--external_packet_csv", default=None)
    parser.add_argument("--external_flow_csv", default=None)

    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)

    # SQLite storage，便于中断后恢复
    parser.add_argument("--storage", default=None)
    parser.add_argument("--study_name", default="stage1_transformer_hpo")

    # 新增：HPO 加速相关参数
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save_trial_config", action="store_true")
    parser.add_argument("--empty_cache_every", type=int, default=10)

    return parser.parse_args()


def infer_input_dim(cfg: Dict[str, Any], preprocessor) -> int:
    """
    复用你 run_stage1.py 里的 input_dim 判断逻辑。
    """
    flow_fusion_cfg = cfg.get("features", {}).get("flow_fusion", {})
    inject_to_packets = flow_fusion_cfg.get("inject_to_packets", True)
    use_flow_features = flow_fusion_cfg.get("enabled", False)

    if inject_to_packets and use_flow_features:
        cfg["_flow_feature_dim"] = 0
        return preprocessor.input_dim()

    if (not inject_to_packets) and use_flow_features and preprocessor.has_flow_features():
        flow_dim = preprocessor.flow_feature_dim()
        cfg["_flow_feature_dim"] = flow_dim
        return preprocessor.packet_feature_dim()

    cfg["_flow_feature_dim"] = 0
    return preprocessor.packet_feature_dim()


def prepare_cached_sampler_weights(train_dataset):
    labels = np.asarray(train_dataset.labels)
    class_counts = np.bincount(labels, minlength=2)
    class_counts = np.maximum(class_counts, 1)

    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]

    # WeightedRandomSampler 可以接受 Tensor，避免每个 trial 重算
    return torch.as_tensor(sample_weights, dtype=torch.double)


def make_loader_kwargs(num_workers: int):
    kwargs = {
        "num_workers": num_workers,
        "collate_fn": custom_collate_fn,
        "pin_memory": torch.cuda.is_available(),
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return kwargs


def make_trial_loaders(
    base_loaders,
    batch_size: int,
    sample_weights: torch.Tensor,
    num_workers: int = 4,
):
    train_dataset = base_loaders["train"].dataset
    val_dataset = base_loaders["val"].dataset
    test_dataset = base_loaders["test"].dataset

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    common_kwargs = make_loader_kwargs(num_workers)

    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            **common_kwargs,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            **common_kwargs,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            **common_kwargs,
        ),
    }


def suggest_hparams(trial: optuna.Trial, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据你的当前配置设计搜索空间。

    当前 config 是：
      d_model=128, nhead=8, num_layers=2, dim_feedforward=512, dropout=0.3
      epochs=1000, batch_size=64, lr=3e-4, weight_decay=5e-4
    搜索时建议先围绕这些值附近搜索，不要一开始放太宽。
    """
    trial_cfg = copy.deepcopy(cfg)

    # d_model 必须能被 nhead 整除
    model_pair = trial.suggest_categorical(
        "model_pair",
        [
            "64_4",
            "64_8",
            "128_4",
            "128_8",
            "192_6",
            "192_8",
            "256_8",
        ],
    )
    d_model, nhead = [int(x) for x in model_pair.split("_")]

    num_layers = trial.suggest_int("num_layers", 1, 4)

    # Transformer FFN 常见为 2x / 4x / 6x d_model
    ffn_multiplier = trial.suggest_categorical("ffn_multiplier", [2, 4, 6])
    dim_feedforward = d_model * ffn_multiplier

    dropout = trial.suggest_float("dropout", 0.10, 0.45)

    trial_cfg.setdefault("model", {})
    trial_cfg["model"]["d_model"] = d_model
    trial_cfg["model"]["nhead"] = nhead
    trial_cfg["model"]["num_layers"] = num_layers
    trial_cfg["model"]["dim_feedforward"] = dim_feedforward
    trial_cfg["model"]["dropout"] = dropout

    trial_cfg.setdefault("features", {})
    trial_cfg["features"].setdefault("flow_fusion", {})
    trial_cfg["features"]["flow_fusion"]["method"] = "gated"

    trial_cfg.setdefault("training", {})
    trial_cfg["training"]["epochs"] = trial.suggest_int("epochs", 40, 100)
    trial_cfg["training"]["early_stop_patience"] = trial.suggest_int(
        "early_stop_patience", 6, 18
    )

    trial_cfg["training"]["batch_size"] = trial.suggest_categorical(
        "batch_size", [32, 64, 128]
    )

    # 对 Transformer，lr 必须 log scale 搜索
    trial_cfg["training"]["lr"] = trial.suggest_float(
        "lr", 1e-5, 1e-3, log=True
    )

    trial_cfg["training"]["weight_decay"] = trial.suggest_float(
        "weight_decay", 1e-6, 5e-3, log=True
    )

    # 你的任务是二分类且不平衡，FocalLoss 的 gamma 可以搜
    trial_cfg["training"]["focal_gamma"] = trial.suggest_float(
        "focal_gamma", 1.0, 3.0
    )

    trial_cfg["training"]["label_smoothing"] = trial.suggest_float(
        "label_smoothing", 0.0, 0.12
    )

    # 继续以恶意类 f1_label1 为主目标
    trial_cfg["training"]["monitor_metric"] = "f1_label1"
    trial_cfg["training"]["monitor_mode"] = "max"

    return trial_cfg


def main():
    args = parse_args()

    safe_mkdir(args.out_dir)

    base_cfg = load_config(args.config)
    base_cfg["seed"] = int(args.seed)
    set_seed(int(args.seed))

    # 加速 cuDNN
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    # 只 build 一次 dataloaders / preprocessor。
    # build_dataloaders 内部已经按 train split fit preprocessor，并生成 npz。
    base_data_dir = os.path.join(args.out_dir, "base_data")

    base_loaders, preprocessor, metadata = build_dataloaders(
        packet_csv=args.packet_csv,
        flow_csv=args.flow_csv,
        cfg=base_cfg,
        out_dir=base_data_dir,
        external_packet_csv=args.external_packet_csv,
        external_flow_csv=args.external_flow_csv,
    )

    input_dim = infer_input_dim(base_cfg, preprocessor)

    print("[INFO] input_dim:", input_dim)
    # print("[INFO] metadata:", json.dumps(metadata, indent=2, ensure_ascii=False))
    print("[INFO] metadata:external_test", metadata["external_test"])
    print("[INFO] metadata:num_train_flows", metadata["num_train_flows"])
    print("[INFO] metadata:num_val_flows", metadata["num_val_flows"])
    print("[INFO] metadata:num_test_flows", metadata["num_test_flows"])
    # 从 metadata 中获取真实的类别计数)
    print("[INFO] metadata:真实训练集类别分布 label_counts_train", metadata["label_counts_train"])
    print("[INFO] metadata:label_counts_val", metadata["label_counts_val"])
    print("[INFO] metadata:label_counts_test", metadata["label_counts_test"])
    print("[INFO] metadata:preprocessor", metadata["preprocessor"])

    save_json(metadata, os.path.join(args.out_dir, "base_metadata.json"))

    # 关键优化：只算一次 sample_weights
    sample_weights = prepare_cached_sampler_weights(base_loaders["train"].dataset)

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = suggest_hparams(trial, base_cfg)

        # infer_input_dim 会写入 _flow_feature_dim。
        # 由于 trial_cfg 是 deepcopy，需要同步 flow_feature_dim。
        _ = infer_input_dim(trial_cfg, preprocessor)

        trial_cfg.setdefault("training", {})
        trial_cfg["training"]["num_workers"] = int(args.num_workers)
        trial_cfg["training"]["amp"] = bool(args.amp)

        batch_size = int(trial_cfg["training"]["batch_size"])

        loaders = make_trial_loaders(
            base_loaders=base_loaders,
            batch_size=batch_size,
            sample_weights=sample_weights,
            num_workers=int(args.num_workers),
        )

        trial_out_dir = os.path.join(args.out_dir, f"trial_{trial.number:04d}")
        safe_mkdir(trial_out_dir)

        # 默认不写每个 trial config，减少 I/O
        if args.save_trial_config:
            with open(
                os.path.join(trial_out_dir, "trial_config.yaml"),
                "w",
                encoding="utf-8",
            ) as f:
                yaml.safe_dump(
                    trial_cfg,
                    f,
                    allow_unicode=True,
                    sort_keys=False,
                )

        set_seed(int(args.seed) + trial.number)

        model = Stage1TimeAwareTransformer(
            input_dim=input_dim,
            cfg=trial_cfg,
        ).to(device)

        try:
            result = train_model_for_optuna(
                model=model,
                loaders=loaders,
                cfg=trial_cfg,
                out_dir=trial_out_dir,
                device=device,
                train_labels=metadata["train_labels"],
                trial=trial,
                save_best=False,
            )

            best_score = float(result["best_score"])

            trial.set_user_attr("best_epoch", int(result["best_epoch"]))
            trial.set_user_attr("best_val_metrics", result["best_val_metrics"])

            return best_score

        finally:
            del model
            del loaders
            gc.collect()

            # 不要每个 trial 都 empty_cache，否则会拖慢
            if (
                torch.cuda.is_available()
                and args.empty_cache_every > 0
                and (trial.number + 1) % args.empty_cache_every == 0
            ):
                torch.cuda.empty_cache()

    sampler = optuna.samplers.TPESampler(
        seed=int(args.seed),
        multivariate=True,
        group=True,
        n_startup_trials=8,
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=8,
        n_warmup_steps=3,
        interval_steps=1,
    )

    storage = args.storage
    if storage is None:
        storage = f"sqlite:///{os.path.join(args.out_dir, 'optuna_study.db')}"

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    timeout = None if int(args.timeout) <= 0 else int(args.timeout)

    study.optimize(
        objective,
        n_trials=int(args.n_trials),
        timeout=timeout,
        gc_after_trial=False,
    )

    print("\n" + "=" * 80)
    print("[OPTUNA] Best trial")
    print("=" * 80)
    print("best_trial.number:", study.best_trial.number)
    print("best_value:", study.best_value)
    print("best_params:", study.best_trial.params)
    print("best_user_attrs:", study.best_trial.user_attrs)

    # 用 best trial 参数重建最佳 config
    best_cfg = suggest_hparams(study.best_trial, base_cfg)
    _ = infer_input_dim(best_cfg, preprocessor)

    best_cfg_path = os.path.join(args.out_dir, "best_config.yaml")
    with open(best_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f, allow_unicode=True, sort_keys=False)

    save_json(
        {
            "best_trial_number": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_trial.params,
            "best_user_attrs": study.best_trial.user_attrs,
            "best_config_path": best_cfg_path,
        },
        os.path.join(args.out_dir, "optuna_best_summary.json"),
    )

    # trials dataframe
    df = study.trials_dataframe()
    df.to_csv(os.path.join(args.out_dir, "optuna_trials.csv"), index=False)

    print(f"\n[INFO] saved best config: {best_cfg_path}")
    print(f"[INFO] saved trials csv: {os.path.join(args.out_dir, 'optuna_trials.csv')}")


if __name__ == "__main__":
    main()