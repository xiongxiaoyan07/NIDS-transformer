from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

def load_stage1_outputs(stage1_dir: str, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load Stage1 embeddings and flow-level metadata.

    Required per split npz keys:
        flow_id, label, z

    Preferred metadata keys, either in npz or stage1_flow_metadata.csv:
        flow_start_timestamp_us, source_id, destination_id

    Returns:
        meta_df: one row per flow. Contains _z_row before sorting.
        z_all: concatenated Stage1 embeddings.
    """
    emb_cfg = cfg["data"].get("embedding_files", {})
    timestamp_col = cfg["data"].get("timestamp_col", "flow_start_timestamp_us")
    source_col = cfg["data"].get("source_col", "source_id")
    destination_col = cfg["data"].get("destination_col", "destination_id")

    dfs: List[pd.DataFrame] = []
    zs: List[np.ndarray] = []
    z_offset = 0

    for split in ["train", "val", "test"]:
        filename = emb_cfg.get(split, f"stage1_{split}_embeddings.npz")
        path = os.path.join(stage1_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing Stage1 embedding file: {path}")

        data = np.load(path, allow_pickle=True)
        for key in ["flow_id", "label", "z"]:
            if key not in data:
                raise KeyError(f"{path} missing required key: {key}")

        # 检查重复
        unique_ids, counts = np.unique(data["flow_id"], return_counts=True)
        dup_ids = unique_ids[counts > 1]
        print("[INFO] Stage1 split: ", split)
        if len(dup_ids) > 0:
            total = (counts[counts > 1] - 1).sum()  # 重复的总条目数（排除第一次出现）
            examples = dup_ids[:10].tolist()
            print(f"Duplicated flow_id in {split}. Total duplicated: {total} Examples: {examples}")

        n = len(data["flow_id"])
        z = data["z"].astype(np.float32)
        if z.shape[0] != n:
            raise ValueError(f"{path}: z rows != flow_id count")

        df = pd.DataFrame(
            {
                "flow_id": data["flow_id"].astype(np.int64),
                "label": data["label"].astype(np.int64),
                "split": split,
                "_z_row": np.arange(z_offset, z_offset + n, dtype=np.int64),
            }
        )

        if "split" in data:
            stored = np.array(data["split"]).astype(str)
            if len(stored) == n and not np.all(stored == split):
                print(f"[WARNING] {path} has split values different from filename split={split}")

        if timestamp_col in data:
            df[timestamp_col] = data[timestamp_col].astype(np.int64)
        if source_col in data:
            df[source_col] = data[source_col].astype(str)
        if destination_col in data:
            df[destination_col] = data[destination_col].astype(str)

        dfs.append(df)
        zs.append(z)
        z_offset += n

    meta_df = pd.concat(dfs, axis=0).reset_index(drop=True)
    z_all = np.concatenate(zs, axis=0).astype(np.float32)

    if meta_df["flow_id"].duplicated().any():
        dup_mask = meta_df["flow_id"].duplicated()
        examples = meta_df.loc[dup_mask, "flow_id"].head(10).tolist()
        total = int(dup_mask.sum())
        print(f"Duplicated flow_id in Stage1 outputs. Total duplicated: {total} Examples: {examples}")

    meta_df = _merge_optional_metadata_csv(meta_df, stage1_dir, cfg)
    _validate_metadata(meta_df, cfg)
    return meta_df, z_all

def _merge_optional_metadata_csv(
    meta_df: pd.DataFrame,
    stage1_dir: str,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    metadata_csv = cfg["data"].get("metadata_csv")
    if not metadata_csv:
        return meta_df

    meta_path = os.path.join(stage1_dir, metadata_csv)
    if not os.path.exists(meta_path):
        return meta_df

    timestamp_col = cfg["data"].get("timestamp_col", "flow_start_timestamp_us")
    source_col = cfg["data"].get("source_col", "source_id")
    destination_col = cfg["data"].get("destination_col", "destination_id")

    raw_meta = pd.read_csv(meta_path)
    candidate_cols = ["flow_id", timestamp_col, source_col, destination_col]
    present_cols = [c for c in candidate_cols if c in raw_meta.columns]
    raw_meta = raw_meta[present_cols].drop_duplicates("flow_id")

    before_cols = set(meta_df.columns)
    meta_df = meta_df.merge(raw_meta, on="flow_id", how="left", suffixes=("", "_csv"))

    for col in [timestamp_col, source_col, destination_col]:
        csv_col = f"{col}_csv"
        if csv_col in meta_df.columns:
            if col not in before_cols:
                meta_df[col] = meta_df[csv_col]
            else:
                meta_df[col] = meta_df[col].where(meta_df[col].notna(), meta_df[csv_col])
            meta_df = meta_df.drop(columns=[csv_col])

    return meta_df

def _validate_metadata(meta_df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    timestamp_col = cfg["data"].get("timestamp_col", "flow_start_timestamp_us")
    source_col = cfg["data"].get("source_col", "source_id")
    destination_col = cfg["data"].get("destination_col", "destination_id")

    required_cols = [
        "flow_id",
        "label",
        "split",
        "_z_row",
        timestamp_col,
        source_col,
        destination_col,
    ]
    missing = [c for c in required_cols if c not in meta_df.columns]
    if missing:
        raise ValueError(
            f"Stage2 missing required metadata columns: {missing}. "
            "Regenerate Stage1 embeddings with timestamp/source/destination metadata, "
            "or provide stage1_flow_metadata.csv."
        )

    meta_df[timestamp_col] = pd.to_numeric(meta_df[timestamp_col], errors="coerce")
    if meta_df[timestamp_col].isna().any():
        bad = int(meta_df[timestamp_col].isna().sum())
        raise ValueError(f"{timestamp_col} contains {bad} missing/non-numeric values")

    if meta_df["flow_id"].duplicated().any():
        examples = meta_df.loc[meta_df["flow_id"].duplicated(), "flow_id"].head(10).tolist()
        raise ValueError(f"Duplicated flow_id in Stage1 outputs. Examples: {examples}")


def prepare_sorted_stage2_data(
    stage1_dir: str,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load Stage1 outputs, sort flows chronologically, and align z rows."""
    meta_df, z_all = load_stage1_outputs(stage1_dir, cfg)
    timestamp_col = cfg["data"].get("timestamp_col", "flow_start_timestamp_us")
    source_col = cfg["data"].get("source_col", "source_id")
    destination_col = cfg["data"].get("destination_col", "destination_id")

    meta_df[timestamp_col] = meta_df[timestamp_col].astype(np.int64)
    meta_df[source_col] = meta_df[source_col].astype(str)
    meta_df[destination_col] = meta_df[destination_col].astype(str)
    meta_df["flow_id"] = meta_df["flow_id"].astype(np.int64)
    meta_df["label"] = meta_df["label"].astype(np.int64)

    meta_df = meta_df.sort_values([timestamp_col, "flow_id"], kind="mergesort").reset_index(drop=True)
    # 它按照排序后的 meta_df["_z_row"] 去重新排列 z_all
    z_sorted = z_all[meta_df["_z_row"].to_numpy(dtype=np.int64)]
    meta_df = meta_df.drop(columns=["_z_row"])
    return meta_df, z_sorted.astype(np.float32)