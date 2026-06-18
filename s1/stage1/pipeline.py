"""
Pipeline builder:
- read CSVs
- split data
- fit preprocessor
- create PyTorch DataLoaders
- support external final test files
"""

from __future__ import annotations

import os
from typing import Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data_io import read_stage1_csvs, get_or_generate_stage1_tensors
from .dataset import PrecomputedFlowDataset
from .preprocessing import Stage1Preprocessor
from .splits import (
    stratified_train_val_test_split,
    train_val_split_for_external_test,
    chronological_train_val_test_split,
    chronological_train_val_split_for_external_test,
)
from .utils import safe_mkdir, save_json, worker_init_fn
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.dataloader import default_collate


def custom_collate_fn(batch):
    """
    自定义 collate 函数，处理可能为 None 的字段
    保持 flow_feats 的 None 语义
    """
    # 检查 batch 中 flow_feats 是否都是 None
    has_flow_feats = any(item.get('flow_feats') is not None for item in batch)

    if has_flow_feats:
        # 如果有 flow_feats，正常处理
        return default_collate(batch)
    else:
        # 如果没有 flow_feats，移除这个字段后再 collate
        batch_without_flow = []
        for item in batch:
            item_copy = {k: v for k, v in item.items() if k != 'flow_feats'}
            batch_without_flow.append(item_copy)
        return default_collate(batch_without_flow)

def build_dataloaders(
    packet_csv: str,
    flow_csv: str,
    cfg: Dict[str, Any],
    out_dir: str,
    external_packet_csv: Optional[str] = None,
    external_flow_csv: Optional[str] = None,
    seed: int = 42
) -> Tuple[Dict[str, DataLoader], Stage1Preprocessor, Dict[str, Any]]:
    """
    Build train/val/test DataLoaders.

    If external test CSVs are given:
        - train/val come from packet_csv/flow_csv
        - test comes from external_packet_csv/external_flow_csv
    Otherwise:
        - train/val/test are split from packet_csv/flow_csv
    """

    print("[INFO] pipeline.py ------ build_dataloaders --- start")

    safe_mkdir(out_dir)

    # seed = int(cfg.get("seed", 42))

    data_cfg = cfg.get("data", {})
    split_cfg = cfg.get("split", {})
    train_cfg = cfg.get("training", {})
    seq_cfg = cfg.get("sequence", {})

    flow_id_col = data_cfg.get("flow_id_col", "flow_id")
    label_col = data_cfg.get("label_col", "label")
    packet_time_col = data_cfg.get("packet_time_col", "timestamp_us")

    max_seq_len = int(seq_cfg.get("max_seq_len", 64))
    strategy = seq_cfg.get("strategy", "head")
    seed = int(cfg.get("seed", 42))

    split_method = split_cfg.get("method", "stratified")
    time_col = data_cfg.get("flow_time_col", "flow_start_timestamp_us")
    print("[INFO] pipeline.py ------ build_dataloaders --- split_method", split_method)

    packets, flows = read_stage1_csvs(
        packet_csv=packet_csv,
        flow_csv=flow_csv,
        flow_id_col=flow_id_col,
        label_col=label_col,
        packet_time_col=packet_time_col,
        max_seq_len = max_seq_len,
        strategy = strategy,
        seed = seed
    )

    has_external_test = external_packet_csv is not None and external_flow_csv is not None

    if has_external_test:
        test_packets, test_flows = read_stage1_csvs(
            packet_csv=external_packet_csv,
            flow_csv=external_flow_csv,
            flow_id_col=flow_id_col,
            label_col=label_col,
            packet_time_col=packet_time_col,
        )

        if split_method == "chronological":
            splits = chronological_train_val_split_for_external_test(
                flows=flows,
                flow_id_col=flow_id_col,
                label_col=label_col,
                time_col=time_col,
                train_size=float(split_cfg.get("train_size", 0.70)),
                val_size=float(split_cfg.get("val_size", 0.10)),
                seed=seed,
                stratify=bool(split_cfg.get("stratify", True)),
                boundary_tolerance=float(split_cfg.get("boundary_tolerance", 0.05)),
            )
        else:
            splits = train_val_split_for_external_test(
                flows=flows,
                flow_id_col=flow_id_col,
                label_col=label_col,
                train_size=float(split_cfg.get("train_size", 0.70)),
                val_size=float(split_cfg.get("val_size", 0.10)),
                seed=seed,
                stratify=bool(split_cfg.get("stratify", True)),
            )

        splits["test"] = [
            int(x) for x in test_flows[flow_id_col].drop_duplicates().tolist()
        ]
    else:
        test_packets, test_flows = packets, flows

        if split_method == "chronological":
            splits = chronological_train_val_test_split(
                flows=flows,
                flow_id_col=flow_id_col,
                label_col=label_col,
                time_col=time_col,
                train_size=float(split_cfg.get("train_size", 0.70)),
                val_size=float(split_cfg.get("val_size", 0.10)),
                test_size=float(split_cfg.get("test_size", 0.20)),
                seed=seed,
                stratify=bool(split_cfg.get("stratify", True)),
                boundary_tolerance=float(split_cfg.get("boundary_tolerance", 0.05)),
            )
        else:
            splits = stratified_train_val_test_split(
                flows=flows,
                flow_id_col=flow_id_col,
                label_col=label_col,
                train_size=float(split_cfg.get("train_size", 0.70)),
                val_size=float(split_cfg.get("val_size", 0.10)),
                test_size=float(split_cfg.get("test_size", 0.20)),
                seed=seed,
                stratify=bool(split_cfg.get("stratify", True)),
            )

    train_ids = set(splits["train"])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])

    train_packets = packets[packets[flow_id_col].isin(train_ids)].copy()
    train_flows = flows[flows[flow_id_col].isin(train_ids)].copy()

    val_packets = packets[packets[flow_id_col].isin(val_ids)].copy()
    val_flows = flows[flows[flow_id_col].isin(val_ids)].copy()

    test_packets_sub = test_packets[test_packets[flow_id_col].isin(test_ids)].copy()
    test_flows_sub = test_flows[test_flows[flow_id_col].isin(test_ids)].copy()

    preprocessor = Stage1Preprocessor(cfg)
    preprocessor.fit(train_packets, train_flows)

    # ===== 新增：设置flow特征维度到config =====
    flow_fusion_cfg = cfg.get("features", {}).get("flow_fusion", {})
    if flow_fusion_cfg.get("enabled", False) and not flow_fusion_cfg.get("inject_to_packets", True):
        # 方案C模式：需要知道flow特征维度
        cfg["_flow_feature_dim"] = preprocessor.flow_feature_dim()
        print(f"[INFO] 方案C模式：flow特征维度={cfg['_flow_feature_dim']}")

    # 在 build_dataloaders 内
    save_dir = os.path.join(out_dir, "precomputed")

    # train
    # train_npz = get_or_generate_stage1_tensors(
    #     train_packets, train_flows, splits["train"], preprocessor, cfg, save_dir, "train"
    # )
    train_npz = get_or_generate_stage1_tensors(
        packets=train_packets,
        flows=train_flows,
        flow_ids=splits["train"],
        preprocessor=preprocessor,
        cfg=cfg,
        out_dir=save_dir,
        prefix="train",
    )

    # val
    # val_npz = get_or_generate_stage1_tensors(
    #     train_packets, train_flows, splits["val"], preprocessor, cfg, save_dir, "val"
    # )
    val_npz = get_or_generate_stage1_tensors(
        packets=val_packets,
        flows=val_flows,
        flow_ids=splits["val"],
        preprocessor=preprocessor,
        cfg=cfg,
        out_dir=save_dir,
        prefix="val",
    )

    # test
    # test_npz = get_or_generate_stage1_tensors(
    #     test_packets, test_flows, splits["test"], preprocessor, cfg, save_dir, "test"
    # )
    test_npz = get_or_generate_stage1_tensors(
        packets=test_packets_sub,
        flows=test_flows_sub,
        flow_ids=splits["test"],
        preprocessor=preprocessor,
        cfg=cfg,
        out_dir=save_dir,
        prefix="test",
    )

    datasets = {
        "train": PrecomputedFlowDataset(train_npz),
        "val": PrecomputedFlowDataset(val_npz),
        "test": PrecomputedFlowDataset(test_npz),
    }

    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 0))

    # 在创建train loader前添加
    train_dataset = datasets["train"]
    train_labels = train_dataset.labels

    # 计算样本权重
    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    print("计算样本权重: class_counts", class_counts)
    print("计算样本权重: sample_weights", sample_weights)

    # 固定种子
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=g
    )
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            sampler=sampler,  # 使用sampler而不是shuffle
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,  # ⚠️ 设置 worker 种子
            collate_fn=custom_collate_fn,
            pin_memory=True,
        ),
        # 修正在保存stage2需要的z和meta info时，在train中出现重复flow问题，这里和run_stage1.py中loader=loaders["trainNoSampler"], 相对应（line261）
        "trainNoSampler": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,  # ⚠️ 设置 worker 种子
            collate_fn=custom_collate_fn,
            pin_memory=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,  # ⚠️ 设置 worker 种子
            collate_fn=custom_collate_fn,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,  # ⚠️ 设置 worker 种子
            collate_fn=custom_collate_fn,
            pin_memory=True,
        ),
    }

    metadata = {
        "external_test": bool(has_external_test),
        "num_train_flows": len(splits["train"]),
        "num_val_flows": len(splits["val"]),
        "num_test_flows": len(splits["test"]),
        "label_counts_train": _label_counts(flows, flow_id_col, label_col, train_ids),
        "label_counts_val": _label_counts(flows, flow_id_col, label_col, val_ids),
        "label_counts_test": _label_counts(test_flows, flow_id_col, label_col, test_ids),
        "splits": splits,
        "preprocessor": preprocessor.summary(),
        "train_labels": train_labels.tolist() if isinstance(train_labels, np.ndarray) else train_labels,
    }

    save_json(metadata, os.path.join(out_dir, "stage1_metadata.json"))
    joblib.dump(preprocessor, os.path.join(out_dir, "stage1_preprocessor.joblib"))

    print("[INFO] pipeline.py ------ build_dataloaders ----- end")

    return loaders, preprocessor, metadata


def _label_counts(df: pd.DataFrame, flow_id_col: str, label_col: str, ids: set) -> Dict[str, int]:
    sub = df[df[flow_id_col].isin(ids)]
    counts = sub[label_col].astype(int).value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}
