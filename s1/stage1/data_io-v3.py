# data_io.py
"""
CSV loading, cleaning and per-flow packet truncation.
"""

from __future__ import annotations

from typing import Dict, Tuple, List, Literal
import numpy as np
import pandas as pd


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

    return packets, flows


def _ensure_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")