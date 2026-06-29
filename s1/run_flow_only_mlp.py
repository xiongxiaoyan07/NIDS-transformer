#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
E2: Flow-only MLP baseline for Stage1.

Purpose:
    Train a pure flow-level MLP using only flow_feats from stage1_flows.csv.
    It does NOT use packet sequence information for classification.

Example:
python /content/drive/MyDrive/s1/run_flow_only_mlp.py \
  --packet_csv /content/drive/MyDrive/s1/data/ar002_et12_20260511_002-stage1_packets.csv \
  --flow_csv /content/drive/MyDrive/s1/data/ar002_et12_20260511_002-stage1_flows.csv \
  --config /content/drive/MyDrive/s1/0629_C/full_gated/stage1_config_0625_C_full_gated.yaml \
  --out_dir /content/drive/MyDrive/s1/0629_C/flow_only_mlp/
"""

from __future__ import annotations

import argparse
import os
import copy

import torch

from stage1.config import load_config
from stage1.pipeline import build_dataloaders
from stage1.baselines import FlowLevelMLP
from stage1.trainer import train_model
from stage1.utils import set_seed, safe_mkdir


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--packet_csv", required=True)
    parser.add_argument("--flow_csv", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--external_packet_csv", default=None)
    parser.add_argument("--external_flow_csv", default=None)

    parser.add_argument("--seed", type=int, default=None)

    # MLP hyperparameters
    parser.add_argument("--hidden_dims", default="256,128,64")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no_batch_norm", action="store_true")

    return parser.parse_args()


def parse_hidden_dims(s: str):
    if s is None or str(s).strip() == "":
        return [256, 128, 64]
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def force_scheme_c_for_flow_feats(cfg: dict) -> dict:
    """
    强制使用方案C的数据生成方式：
        x: packet-only tensor
        flow_feats: separate flow-level tensor

    Flow-only MLP 只会使用 flow_feats。
    """
    cfg = copy.deepcopy(cfg)

    cfg.setdefault("features", {})
    cfg["features"].setdefault("flow_fusion", {})

    cfg["features"]["flow_fusion"]["enabled"] = True
    cfg["features"]["flow_fusion"]["inject_to_packets"] = False

    # method 对 FlowLevelMLP 本身没有影响，只是为了日志清楚
    cfg["features"]["flow_fusion"]["method"] = "flow_only_mlp"

    return cfg


def get_first_batch_flow_dim(loaders):
    batch = next(iter(loaders["trainNoSampler"]))
    flow_feats = batch.get("flow_feats", None)

    if flow_feats is None:
        raise RuntimeError(
            "flow_feats is None. 请检查 config 中 features.flow_fusion.enabled=True "
            "且 inject_to_packets=False，并确认 flow feature columns 已配置。"
        )

    if flow_feats.ndim != 2:
        raise RuntimeError(
            f"flow_feats should be [B, D_flow], got shape={tuple(flow_feats.shape)}"
        )

    return int(flow_feats.shape[1])


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def main():
    args = parse_args()

    cfg = load_config(args.config)

    seed = args.seed
    if seed is None:
        seed = int(cfg.get("seed", 42))

    set_seed(seed)

    # 单独目录，避免覆盖 full_gated 的 best model / history / metrics
    safe_mkdir(args.out_dir)

    print("=" * 80)
    print("[E2] Flow-only MLP baseline")
    print("=" * 80)
    print("[E2] packet_csv:", args.packet_csv)
    print("[E2] flow_csv:", args.flow_csv)
    print("[E2] config:", args.config)
    print("[E2] out_dir:", args.out_dir)
    print("[E2] seed:", seed)

    # 关键：强制方案C，让 pipeline 生成单独 flow_feats
    cfg = force_scheme_c_for_flow_feats(cfg)

    print("\n[STEP 1] Building dataloaders with separate flow_feats...")
    loaders, preprocessor, metadata = build_dataloaders(
        packet_csv=args.packet_csv,
        flow_csv=args.flow_csv,
        cfg=cfg,
        out_dir=args.out_dir,
        external_packet_csv=args.external_packet_csv,
        external_flow_csv=args.external_flow_csv,
        seed=seed,
    )

    # 从 preprocessor 或 batch 里拿 flow feature dim
    if hasattr(preprocessor, "flow_feature_dim"):
        flow_feature_dim = int(preprocessor.flow_feature_dim())
    else:
        flow_feature_dim = get_first_batch_flow_dim(loaders)

    # 再做一次 batch 检查，防止 precomputed 缓存错误
    batch_flow_dim = get_first_batch_flow_dim(loaders)
    if batch_flow_dim != flow_feature_dim:
        raise RuntimeError(
            f"flow_feature_dim mismatch: preprocessor={flow_feature_dim}, "
            f"batch={batch_flow_dim}"
        )

    num_classes = int(cfg.get("model", {}).get("num_classes", 2))
    hidden_dims = parse_hidden_dims(args.hidden_dims)

    print("\n[STEP 2] Building Flow-only MLP...")
    print("[INFO] E2 - Flow-only MLP")
    print(f"[INFO]   Flow feature dim: {flow_feature_dim}")
    print(f"[INFO]   Hidden dims: {hidden_dims}")
    print(f"[INFO]   Dropout: {args.dropout}")
    print(f"[INFO]   BatchNorm: {not args.no_batch_norm}")
    print(f"[INFO]   Num classes: {num_classes}")

    model = FlowLevelMLP(
        input_dim=flow_feature_dim,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        num_classes=num_classes,
        use_batch_norm=not args.no_batch_norm,
    )

    total_params, trainable_params = count_parameters(model)
    print(f"[INFO]   Total params: {total_params:,}")
    print(f"[INFO]   Trainable params: {trainable_params:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    # train_model 需要 train_labels，用 metadata 里已有的即可
    train_labels = metadata.get("train_labels", None)
    if train_labels is None:
        train_labels = []
        for batch in loaders["trainNoSampler"]:
            train_labels.extend(batch["label"].numpy().tolist())

    print("\n[STEP 3] Training Flow-only MLP...")
    results = train_model(
        model=model,
        loaders=loaders,
        cfg=cfg,
        out_dir=args.out_dir,
        device=device,
        train_labels=train_labels,
    )

    print("\n[E2 DONE] Flow-only MLP finished.")
    print("[E2] Best model:", results.get("best_model_path"))
    print("[E2] Best epoch:", results.get("best_epoch"))
    print("[E2] Best score:", results.get("best_score"))


if __name__ == "__main__":
    main()

# !rm - rf / content / drive / MyDrive / s1 / 0629_C / flow_only_mlp / precomputed
#
# !python / content / drive / MyDrive / s1 / run_flow_only_mlp.py \
#  --packet_csv / content / drive / MyDrive / s1 / data / ar002_et12_20260511_002 - stage1_packets.csv \
#  --flow_csv / content / drive / MyDrive / s1 / data / ar002_et12_20260511_002 - stage1_flows.csv \
#  --config / content / drive / MyDrive / s1 / 0629_C / full_gated / stage1_config_0625_C_full_gated.yaml \
#  --out_dir / content / drive / MyDrive / s1 / 0629_C / flow_only_mlp /