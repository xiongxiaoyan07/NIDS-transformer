"""
CSV loading and cleaning.
"""

from __future__ import annotations

from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Strip column names.
    - Drop Suricata record_type if present.
    - Replace inf/-inf with NaN.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "record_type" in df.columns:
        df = df.drop(columns=["record_type"])

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def read_stage1_csvs(
    packet_csv: str,
    flow_csv: str,
    flow_id_col: str,
    label_col: str,
    packet_time_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read and clean stage1_packets.csv and stage1_flows.csv.
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
    packet_ids = set(packets[flow_id_col].unique().tolist())
    flow_ids = set(flows[flow_id_col].unique().tolist())
    common_ids = packet_ids.intersection(flow_ids)

    packets = packets[packets[flow_id_col].isin(common_ids)].copy()
    flows = flows[flows[flow_id_col].isin(common_ids)].copy()

    if len(flows) == 0:
        raise ValueError("No common flow_id found between packet CSV and flow CSV.")

    return packets, flows


def _ensure_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
