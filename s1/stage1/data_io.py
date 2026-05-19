# data_io.py
"""
CSV loading, cleaning and per-flow packet truncation.
"""

from __future__ import annotations

from typing import Dict, Tuple, List, Literal
import numpy as np
import pandas as pd
import os
import torch
import joblib
from scipy.signal import max_len_seq

from .preprocessing import Stage1Preprocessor

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Strip column names.
    - Drop Suricata record_type if present.
    - Replace inf/-inf with NaN.
    - Remove flow_id == 0
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "record_type" in df.columns:
        df = df.drop(columns=["record_type"])

    df = df.replace([np.inf, -np.inf], np.nan)
    # 去除掉flow_id为0的数据
    df = df[df['flow_id'] != 0]

    return df


def read_stage1_csvs(
    packet_csv: str,
    flow_csv: str,
    flow_id_col: str,
    label_col: str,
    packet_time_col: str,
    strategy: str = "head",
    max_seq_len: int = 64,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read and clean stage1_packets.csv and stage1_flows.csv.
    Apply per-flow packet truncation to save memory.
    """

    print("[INFO] data_io.py ------ read_stage1_csvs --- start")

    packets = clean_dataframe(pd.read_csv(packet_csv))
    flows = clean_dataframe(pd.read_csv(flow_csv))

    required_packets = [flow_id_col, packet_time_col]
    required_flows = [flow_id_col, label_col]

    _ensure_columns(packets, required_packets, "packet csv")
    _ensure_columns(flows, required_flows, "flow csv")

    packets[flow_id_col] = pd.to_numeric(
        packets[flow_id_col], errors="coerce"
    ).fillna(-1).astype("int64")

    flows[flow_id_col] = pd.to_numeric(
        flows[flow_id_col], errors="coerce"
    ).fillna(-1).astype("int64")

    flows[label_col] = pd.to_numeric(
        flows[label_col], errors="coerce"
    ).fillna(0).astype(int)

    packets[packet_time_col] = pd.to_numeric(
        packets[packet_time_col], errors="coerce"
    ).fillna(0).astype("int64")

    # Keep only flow IDs that exist in both files.
    # packet_ids = set(packets[flow_id_col].unique().tolist())
    flow_ids = set(flows[flow_id_col].unique().tolist())
    packets = packets[packets[flow_id_col].isin(flow_ids)].copy()
    flows = flows[flows[flow_id_col].isin(flow_ids)].copy()

    # ---------- 按 flow 截取 packet ----------
    if strategy not in {"head", "head_tail", "random"}:
        raise ValueError("sequence.strategy must be one of: head, head_tail, random")

    packets_list = []

    for fid, group in packets.groupby(flow_id_col, sort=False):
        df = group.sort_values(packet_time_col)
        n = len(df)

        if n <= max_seq_len:
            packets_list.append(df)
            continue

        if strategy == "head":
            packets_list.append(df.iloc[:max_seq_len])
            continue

        if strategy == "head_tail":
            half = max_seq_len // 2
            first = df.iloc[:half]
            last = df.iloc[-(max_seq_len - half):]
            packets_list.append(pd.concat([first, last], axis=0).sort_values(packet_time_col))
            continue

        # random
        rng = np.random.default_rng(seed + int(fid) % 1000003)
        chosen = rng.choice(n, size=max_seq_len, replace=False)
        chosen = np.sort(chosen)
        packets_list.append(df.iloc[chosen].sort_values(packet_time_col))

    packets = pd.concat(packets_list, axis=0)

    print("[INFO] data_io.py ------ read_stage1_csvs --- end")

    return packets, flows


def _ensure_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")

def generate_and_save_stage1_tensors(
    packets: pd.DataFrame,
    flows: pd.DataFrame,
    flow_ids: list,
    preprocessor: Stage1Preprocessor,
    cfg: dict,
    out_dir: str,
    save_name_prefix: str = "stage1_precomputed",
):
    """
    For each flow, precompute:
        x: [max_seq_len, feature_dim]
        time: [max_seq_len]
        mask: [max_seq_len]
        label: scalar

    Save as .npz for reuse.
    """
    print("[INFO] data_io.py ------ generate_and_save_stage1_tensors --- start")

    os.makedirs(out_dir, exist_ok=True)
    seq_cfg = cfg.get("sequence", {})
    max_seq_len = int(seq_cfg.get("max_seq_len", 64))
    strategy = seq_cfg.get("strategy", "head")
    packet_iat_col = cfg["data"]["packet_iat_col"]
    flow_id_col = cfg["data"]["flow_id_col"]
    label_col = cfg["data"]["label_col"]

    feature_dim = preprocessor.input_dim()
    num_flows = len(flow_ids)

    x_tensor = np.zeros((num_flows, max_seq_len, feature_dim), dtype=np.float32)
    time_tensor = np.zeros((num_flows, max_seq_len), dtype=np.float32)
    mask_tensor = np.zeros((num_flows, max_seq_len), dtype=bool)
    labels = np.zeros((num_flows,), dtype=np.int64)
    flow_id_tensor = np.zeros((num_flows,), dtype=np.int64)

    flow_rows = {int(row[flow_id_col]): row for _, row in flows.iterrows()}

    for idx, fid in enumerate(flow_ids):
        pkt_df = packets[packets[flow_id_col] == fid].sort_values(cfg["data"]["packet_time_col"])
        flow_df = pd.DataFrame([flow_rows[fid]])

        packet_x = preprocessor.transform_packets(pkt_df)
        flow_x = preprocessor.transform_flows(flow_df)
        flow_x_tiled = np.repeat(flow_x, repeats=len(pkt_df), axis=0)
        x = np.concatenate([packet_x, flow_x_tiled], axis=1).astype(np.float32)

        # time log
        if packet_iat_col in pkt_df.columns:
            time_raw = pd.to_numeric(pkt_df[packet_iat_col], errors="coerce").fillna(0).clip(lower=0).to_numpy(dtype=np.float32)
        else:
            time_raw = np.zeros(len(pkt_df), dtype=np.float32)
        time_log = np.log1p(time_raw).astype(np.float32)

        real_len = min(len(pkt_df), max_seq_len)
        x_tensor[idx, :real_len] = x[:real_len]
        time_tensor[idx, :real_len] = time_log[:real_len]
        mask_tensor[idx, :real_len] = True
        labels[idx] = int(flow_rows[fid][label_col])
        flow_id_tensor[idx] = fid

    save_path = os.path.join(out_dir, f"seqLen{max_seq_len}{strategy}{save_name_prefix}.npz")
    np.savez_compressed(save_path,
                        x=x_tensor,
                        time=time_tensor,
                        mask=mask_tensor,
                        labels=labels,
                        flow_ids=flow_id_tensor)
    print(f"[INFO] saved precomputed stage1 tensors: {save_path}")
    return save_path

def get_or_generate_stage1_tensors(
    packets: pd.DataFrame,
    flows: pd.DataFrame,
    flow_ids: List[int],
    preprocessor: Stage1Preprocessor,
    cfg: dict,
    out_dir: str,
    prefix: str
) -> str:
    """
    If precomputed .npz exists, return its path.
    Otherwise, generate and save.
    """
    os.makedirs(out_dir, exist_ok=True)
    seq_cfg = cfg.get("sequence", {})
    max_seq_len = int(seq_cfg.get("max_seq_len", 64))
    strategy = seq_cfg.get("strategy", "head")
    npz_path = os.path.join(out_dir, f"seqLen{max_seq_len}{strategy}{prefix}.npz")
    if os.path.exists(npz_path):
        print(f"[INFO] found existing precomputed tensor: {npz_path}")
        return npz_path

    npz_path = generate_and_save_stage1_tensors(
        packets=packets,
        flows=flows,
        flow_ids=flow_ids,
        preprocessor=preprocessor,
        cfg=cfg,
        out_dir=out_dir,
        save_name_prefix=prefix
    )
    return npz_path