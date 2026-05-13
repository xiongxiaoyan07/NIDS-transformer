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
import pandas as pd
from torch.utils.data import DataLoader

from .data_io import read_stage1_csvs
from .dataset import Stage1FlowDataset
from .preprocessing import Stage1Preprocessor
from .splits import stratified_train_val_test_split, train_val_split_for_external_test
from .utils import safe_mkdir, save_json


def build_dataloaders(
    packet_csv: str,
    flow_csv: str,
    cfg: Dict[str, Any],
    out_dir: str,
    external_packet_csv: Optional[str] = None,
    external_flow_csv: Optional[str] = None,
) -> Tuple[Dict[str, DataLoader], Stage1Preprocessor, Dict[str, Any]]:
    """
    Build train/val/test DataLoaders.

    If external test CSVs are given:
        - train/val come from packet_csv/flow_csv
        - test comes from external_packet_csv/external_flow_csv
    Otherwise:
        - train/val/test are split from packet_csv/flow_csv
    """
    safe_mkdir(out_dir)

    seed = int(cfg.get("seed", 42))

    data_cfg = cfg.get("data", {})
    split_cfg = cfg.get("split", {})
    train_cfg = cfg.get("training", {})

    flow_id_col = data_cfg.get("flow_id_col", "flow_id")
    label_col = data_cfg.get("label_col", "label")
    packet_time_col = data_cfg.get("packet_time_col", "timestamp_us")

    packets, flows = read_stage1_csvs(
        packet_csv=packet_csv,
        flow_csv=flow_csv,
        flow_id_col=flow_id_col,
        label_col=label_col,
        packet_time_col=packet_time_col,
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

    preprocessor = Stage1Preprocessor(cfg)
    preprocessor.fit(train_packets, train_flows)

    datasets = {
        "train": Stage1FlowDataset(
            packets=packets[packets[flow_id_col].isin(train_ids)].copy(),
            flows=flows[flows[flow_id_col].isin(train_ids)].copy(),
            flow_ids=splits["train"],
            preprocessor=preprocessor,
            cfg=cfg,
        ),
        "val": Stage1FlowDataset(
            packets=packets[packets[flow_id_col].isin(val_ids)].copy(),
            flows=flows[flows[flow_id_col].isin(val_ids)].copy(),
            flow_ids=splits["val"],
            preprocessor=preprocessor,
            cfg=cfg,
        ),
        "test": Stage1FlowDataset(
            packets=test_packets[test_packets[flow_id_col].isin(test_ids)].copy(),
            flows=test_flows[test_flows[flow_id_col].isin(test_ids)].copy(),
            flow_ids=splits["test"],
            preprocessor=preprocessor,
            cfg=cfg,
        ),
    }

    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 0))

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
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
    }

    save_json(metadata, os.path.join(out_dir, "stage1_metadata.json"))
    joblib.dump(preprocessor, os.path.join(out_dir, "stage1_preprocessor.joblib"))

    return loaders, preprocessor, metadata


def _label_counts(df: pd.DataFrame, flow_id_col: str, label_col: str, ids: set) -> Dict[str, int]:
    sub = df[df[flow_id_col].isin(ids)]
    counts = sub[label_col].astype(int).value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}
