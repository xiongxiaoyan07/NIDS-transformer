#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entry point for Stage1 training.

Examples:

1) Normal train/val/test split:

python run_stage1.py \
  --packet_csv /home/xxiong/pcaps/stage1_packets.csv \
  --flow_csv /home/xxiong/pcaps/stage1_flows.csv \
  --config configs/stage1_config.yaml \
  --out_dir ./stage1_artifacts

2) External final test:

python run_stage1.py \
  --packet_csv ./train_stage1_packets.csv \
  --flow_csv ./train_stage1_flows.csv \
  --external_packet_csv ./final_test_packets.csv \
  --external_flow_csv ./final_test_flows.csv \
  --config configs/stage1_config.yaml \
  --out_dir ./stage1_artifacts_external_test
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

from stage1.config import load_config
from stage1.model import Stage1TimeAwareTransformer
from stage1.pipeline import build_dataloaders
from stage1.trainer import train_model
from stage1.export_embeddings import export_embeddings
from stage1.utils import set_seed, safe_mkdir, save_json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--packet_csv", required=True)
    parser.add_argument("--flow_csv", required=True)
    parser.add_argument("--config", default="configs/stage1_config.yaml")
    parser.add_argument("--out_dir", default="./stage1_artifacts")

    parser.add_argument("--external_packet_csv", default=None)
    parser.add_argument("--external_flow_csv", default=None)

    parser.add_argument("--export_embeddings", action="store_true")

    return parser.parse_args()

def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def build_stage2_metadata_from_npz(
    npz_path: str,
    split_name: str,
) -> pd.DataFrame:
    print("[INFO] Stage1 build_stage2_metadata_from_npz  npz_path: ", npz_path)
    data = np.load(npz_path, allow_pickle=True)

    required_keys = ["flow_id", "label"]
    missing = [k for k in required_keys if k not in data]

    if missing:
        raise KeyError(f"{npz_path} missing required keys: {missing}")

    # # 检查重复
    # unique_ids, counts = np.unique(data["flow_id"], return_counts=True)
    # dup_ids = unique_ids[counts > 1]
    # print("[INFO] Stage1 build_stage2_metadata_from_npz: ", split_name)
    # # train: 这里也有重复的15547-----修改之后这里就不存在重复的flow_id了
    # if len(dup_ids) > 0:
    #     total = (counts[counts > 1] - 1).sum()  # 重复的总条目数（排除第一次出现）
    #     examples = dup_ids[:10].tolist()
    #     print(f"[INFO] Stage1 build_stage2_metadata_from_npz--------------Duplicated flow_id in {split_name}. Total duplicated: {total} Examples: {examples}")

    n = len(data["flow_id"])

    if len(data["label"]) != n:
        raise ValueError(
            f"{npz_path}: flow_id length={n}, "
            f"label length={len(data['label'])}"
        )

    meta = {
        "flow_id": data["flow_id"].astype(np.int64),
        "label": data["label"].astype(np.int64),
    }

    if "split" in data:
        if len(data["split"]) != n:
            raise ValueError(
                f"{npz_path}: split length={len(data['split'])}, "
                f"flow_id length={n}"
            )
        meta["split"] = data["split"].astype(str)
    else:
        meta["split"] = np.full(n, split_name, dtype=object)

    optional_cols = [
        "flow_start_timestamp_us",
        "source_id",
        "destination_id",
    ]

    for col in optional_cols:
        if col not in data:
            continue

        if len(data[col]) != n:
            raise ValueError(
                f"{npz_path}: {col} length={len(data[col])}, "
                f"flow_id length={n}"
            )

        if col == "flow_start_timestamp_us":
            meta[col] = data[col].astype(np.int64)
        else:
            meta[col] = data[col].astype(str)

    return pd.DataFrame(meta)

def main():
    args = parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))

    set_seed(seed)
    safe_mkdir(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    # Step 1: Build dataloaders and preprocessor
    print("\n" + "=" * 60)
    print("[STEP 1] Building dataloaders...")
    print("=" * 60)

    loaders, preprocessor, metadata = build_dataloaders(
        packet_csv=args.packet_csv,
        flow_csv=args.flow_csv,
        cfg=cfg,
        out_dir=args.out_dir,
        external_packet_csv=args.external_packet_csv,
        external_flow_csv=args.external_flow_csv,
    )
    # Print summary
    print("\n[INFO] Preprocessor Summary:")
    print("external_test", metadata["external_test"])
    print("num_train_flows", metadata["num_train_flows"])
    print("num_val_flows", metadata["num_val_flows"])
    print("num_test_flows", metadata["num_test_flows"])
    # 从 metadata 中获取真实的类别计数)
    print("真实训练集类别分布 label_counts_train", metadata["label_counts_train"])
    print("label_counts_val", metadata["label_counts_val"])
    print("label_counts_test", metadata["label_counts_test"])
    print("preprocessor", metadata["preprocessor"])

    # Step 2: Determine model input dimensions
    print("\n" + "=" * 60)
    print("[STEP 2] Building model...")
    print("=" * 60)
    flow_fusion_cfg = cfg.get("features", {}).get("flow_fusion", {})
    inject_to_packets = flow_fusion_cfg.get("inject_to_packets", True)
    use_flow_features = flow_fusion_cfg.get("enabled", False)

    if inject_to_packets and use_flow_features:
        # 方案A: flow特征拼接到packet
        input_dim = preprocessor.input_dim()
        print(f"[INFO] 方案A - Flow特征拼接到每个Packet")
        print(f"[INFO]   Input dim (with flow): {input_dim}")
        # 不需要flow_feature_dim
        cfg["_flow_feature_dim"] = 0
    elif not inject_to_packets and use_flow_features and preprocessor.has_flow_features():
        # 方案C: 分层特征注入
        input_dim = preprocessor.packet_feature_dim()
        flow_dim = preprocessor.flow_feature_dim()
        cfg["_flow_feature_dim"] = flow_dim
        print(f"[INFO] 方案C - 分层特征注入")
        print(f"[INFO]   Packet input dim: {input_dim}")
        print(f"[INFO]   Flow feature dim: {flow_dim}")
    else:
        # 方案B: 仅packet特征
        input_dim = preprocessor.packet_feature_dim()
        cfg["_flow_feature_dim"] = 0
        print(f"[INFO] 方案B - 仅使用Packet特征")
        print(f"[INFO]   Input dim: {input_dim}")

    # Build model
    model = Stage1TimeAwareTransformer(
        input_dim=input_dim,
        cfg=cfg,
    ).to(device)
    # input_dim = preprocessor.input_dim()
    # model = Stage1TimeAwareTransformer(input_dim=input_dim, cfg=cfg).to(device)

    # 打印参数量
    total_params, trainable_params = count_parameters(model)
    print(f"\n{'=' * 60}")
    print(f"📊 Model Statistics:")
    print(f"{'=' * 60}")
    print(f"  Total parameters:      {total_params:,}")
    print(f"  Trainable parameters:  {trainable_params:,}")
    print(f"  Non-trainable params:  {total_params - trainable_params:,}")

    # 打印各组件参数量
    print(f"\n  Parameter breakdown:")
    print(f"  {'─' * 40}")
    for name, param in model.named_parameters():
        print(f"    {name:<40} {param.numel():>10,}")

    # 如果有GPU，也打印模型大小
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print(f"\n  Model size: {size_all_mb:.2f} MB")
    print(f"{'=' * 60}\n")

    # Step 3: Train
    print("\n" + "=" * 60)
    print("[STEP 3] Training model...")
    print("=" * 60)
    run_summary = train_model(
        model=model,
        loaders=loaders,
        cfg=cfg,
        out_dir=args.out_dir,
        device=device,
        train_labels=metadata["train_labels"]
    )

    save_json(run_summary, os.path.join(args.out_dir, "stage1_run_summary.json"))

    print("\n" + "=" * 60)
    print("[STEP 4] Exporting embeddings...args", args)
    print("=" * 60)
    if args.export_embeddings:
        meta_dfs = []

        for split_name in ["train", "val", "test"]:
            out_path = os.path.join(
                args.out_dir,
                f"stage1_{split_name}_embeddings.npz",
            )
            if split_name == "train":
                export_embeddings(
                    model=model,
                    loader=loaders["trainNoSampler"],
                    device=device,
                    output_npz_path=out_path,
                    split_name=split_name,
                )
            else:
                export_embeddings(
                    model=model,
                    loader=loaders[split_name],
                    device=device,
                    output_npz_path=out_path,
                    split_name=split_name,
                )

            print(f"[INFO] saved embeddings: {out_path}")

            split_meta_df = build_stage2_metadata_from_npz(
                npz_path=out_path,
                split_name=split_name,
            )
            meta_dfs.append(split_meta_df)

        meta_df = pd.concat(meta_dfs, axis=0, ignore_index=True)

        if "flow_start_timestamp_us" in meta_df.columns:
            meta_df = meta_df.sort_values(
                ["flow_start_timestamp_us", "flow_id"],
                kind="mergesort",
            ).reset_index(drop=True)

        meta_path = os.path.join(args.out_dir, "stage1_flow_metadata.csv")
        meta_df.to_csv(meta_path, index=False)

        print(f"[INFO] saved Stage2 flow metadata: {meta_path}")

if __name__ == "__main__":
    main()
