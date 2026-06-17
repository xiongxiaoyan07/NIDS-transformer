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

def clean_dataframe(df: pd.DataFrame, type_str: str) -> pd.DataFrame:
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
    # df = df[df['flow_id'] != 0]
    if type_str == "flow":
        df = df[(df['flow_id'] != 0) & (df['flow_start_timestamp_us'] != 0) &
                df['flow_id'].notna() & df['flow_start_timestamp_us'].notna()]
    else:
        df = df[(df['flow_id'] != 0) & (df['timestamp_us'] != 0) &
                df['flow_id'].notna() & df['timestamp_us'].notna()]

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

    packets = clean_dataframe(pd.read_csv(packet_csv), "packet")
    flows = clean_dataframe(pd.read_csv(flow_csv), "flow")

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
        flow_feats: [flow_feature_dim] (only when inject_to_packets=False)

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
    packet_time_col = cfg["data"]["packet_time_col"]

    flow_time_col = cfg["data"].get("flow_time_col", "flow_start_timestamp_us")
    source_id_col = cfg["data"].get("source_id_col", "source_id")
    destination_id_col = cfg["data"].get("destination_id_col", "destination_id")

    # 读取 flow 融合配置
    flow_fusion_cfg = cfg.get("features", {}).get("flow_fusion", {})
    inject_to_packets = flow_fusion_cfg.get("inject_to_packets", True)
    use_flow_features = flow_fusion_cfg.get("enabled", False)

    # print("[INFO] data_ip.py ------ generate_and_save_stage1_tensors  00000 ---", save_name_prefix)
    # print("[INFO] data_io.py ------ generate_and_save_stage1_tensors --- 11111 flows shape = ", flows.shape)
    # # 这里也是没有重复的
    # if flows["flow_id"].duplicated().any():
    #     dup_mask = flows["flow_id"].duplicated()
    #     examples = flows.loc[dup_mask, "flow_id"].head(10).tolist()
    #     total = int(dup_mask.sum())
    #     print(f"data_io.py------Duplicated flow_id in Stage1 flows. Total duplicated: {total} Examples: {examples}")
    #     # 检查重复
    #
    # unique_ids, counts = np.unique(np.array(flow_ids, dtype=np.int64), return_counts=True)
    # dup_ids = unique_ids[counts > 1]
    # print("[INFO] data_io.py----2222222----flow_ids len: ", len(flow_ids))
    # # 这里也没有重复的
    # if len(dup_ids) > 0:
    #     total = (counts[counts > 1] - 1).sum()  # 重复的总条目数（排除第一次出现）
    #     examples = dup_ids[:10].tolist()
    #     print(
    #         f"[INFO]  data_io.py----2222222---------------Total duplicated: {total} Examples: {examples}")

    # 1. 固定 flow_ids 顺序
    flow_ids = [int(x) for x in flow_ids]
    num_flows = len(flow_ids)

    # 2. 只保留需要的 flow，并按 flow_ids 排序
    flow_order = pd.DataFrame({flow_id_col: flow_ids})
    flows_sub = flow_order.merge(flows, on=flow_id_col, how="left")

    if flows_sub[label_col].isna().any():
        missing = flows_sub.loc[flows_sub[label_col].isna(), flow_id_col].tolist()[:10]
        raise ValueError(f"Some flow_ids are missing in flows. Examples: {missing}")

    labels = flows_sub[label_col].astype(int).to_numpy(dtype=np.int64)
    flow_id_tensor = np.asarray(flow_ids, dtype=np.int64)

    # 3. packets 一次性排序
    packets_sub = packets[packets[flow_id_col].isin(flow_ids)].copy()
    packets_sub = packets_sub.sort_values([flow_id_col, packet_time_col])

    # 4. 一次性转换所有 packet features
    # 注意：这里先只做 packet features，不拼 flow features
    packet_x_all = preprocessor.transform_packets_only(packets_sub).astype(np.float32)
    print("[DEBUG] packet_x_all shape:", packet_x_all.shape)
    print("[DEBUG] packet_x_all sample:\n", packet_x_all[:1, :])  # 打印前5个packet的前10个特征

    # 5. 一次性构造 time
    if packet_iat_col in packets_sub.columns:
        time_raw_all = (
            pd.to_numeric(packets_sub[packet_iat_col], errors="coerce")
            .fillna(0)
            .clip(lower=0)
            .to_numpy(dtype=np.float32)
        )
        time_log_all = np.log1p(time_raw_all).astype(np.float32)
    else:
        time_log_all = np.zeros(len(packets_sub), dtype=np.float32)

    # 如果你想彻底去掉 flow_iat_us 信息，可以改成：
    # time_log_all = np.zeros(len(packets_sub), dtype=np.float32)

    # 6. 一次性转换所有 flow features
    if use_flow_features and preprocessor.has_flow_features():
        flow_x_all = preprocessor.transform_flows(flows_sub).astype(np.float32)
        print("[DEBUG] flow_x_all shape:", flow_x_all.shape)
        print("[DEBUG] flow_x_all sample:\n", flow_x_all[:1, :])
    else:
        flow_x_all = None

    packet_feature_dim = preprocessor.packet_feature_dim()

    if inject_to_packets and use_flow_features and flow_x_all is not None:
        flow_feature_dim = preprocessor.flow_feature_dim()
        feature_dim = packet_feature_dim + flow_feature_dim
        flow_feats_tensor = None
        print("[INFO] 模式: 方案A - Flow特征拼接到Packets")
    elif not inject_to_packets and use_flow_features and flow_x_all is not None:
        flow_feature_dim = preprocessor.flow_feature_dim()
        feature_dim = packet_feature_dim
        flow_feats_tensor = flow_x_all
        print("[INFO] 模式: 方案C - 分层特征注入")
    else:
        feature_dim = packet_feature_dim
        flow_feats_tensor = None
        print("[INFO] 模式: 方案B - 仅Packet特征")

    x_tensor = np.zeros((num_flows, max_seq_len, feature_dim), dtype=np.float32)
    time_tensor = np.zeros((num_flows, max_seq_len), dtype=np.float32)
    mask_tensor = np.zeros((num_flows, max_seq_len), dtype=bool)

    # 7. 用 groupby.indices 避免每个 fid 全表扫描
    group_indices = packets_sub.groupby(flow_id_col, sort=False).indices

    flow_id_to_row = {fid: i for i, fid in enumerate(flow_ids)}

    for fid, row_idx in flow_id_to_row.items():
        indices = group_indices.get(fid)

        if indices is None:
            print("indices is none")
            continue

        real_len = min(len(indices), max_seq_len)
        idxs = indices[:real_len]

        if inject_to_packets and use_flow_features and flow_x_all is not None:
            x_tensor[row_idx, :real_len, :packet_feature_dim] = packet_x_all[idxs]
            x_tensor[row_idx, :real_len, packet_feature_dim:] = flow_x_all[row_idx]
        else:
            x_tensor[row_idx, :real_len] = packet_x_all[idxs]

        time_tensor[row_idx, :real_len] = time_log_all[idxs]
        mask_tensor[row_idx, :real_len] = True

    # ===== Stage2 metadata: keep raw flow-level identifiers =====
    # These fields are NOT used as Stage1 model input.
    # They are saved only for Stage2 inter-flow context construction.
    stage2_meta = {}

    if flow_time_col in flows_sub.columns:
        stage2_meta["flow_start_timestamp_us"] = pd.to_numeric(
            flows_sub[flow_time_col],
            errors="coerce",
        ).fillna(0).astype("int64").to_numpy()
    else:
        raise ValueError(
            f"Missing required Stage2 time column: {flow_time_col}. "
            "Stage2 needs flow_start_timestamp_us for chronological windows."
        )

    if source_id_col in flows_sub.columns:
        stage2_meta["source_id"] = flows_sub[source_id_col].astype(str).to_numpy()
    else:
        raise ValueError(
            f"Missing required Stage2 source column: {source_id_col}."
        )

    if destination_id_col in flows_sub.columns:
        stage2_meta["destination_id"] = flows_sub[destination_id_col].astype(str).to_numpy()
    else:
        raise ValueError(
            f"Missing required Stage2 destination column: {destination_id_col}."
        )

    save_path = os.path.join(
        out_dir,
        f"seqLen{max_seq_len}{strategy}{save_name_prefix}.npz"
    )

    save_dict = {
        "x": x_tensor,
        "time": time_tensor,
        "mask": mask_tensor,
        "labels": labels,
        "flow_ids": flow_id_tensor,
        # Stage2 metadata
        "flow_start_timestamp_us": stage2_meta["flow_start_timestamp_us"],
        "source_id": stage2_meta["source_id"],
        "destination_id": stage2_meta["destination_id"],
    }

    if flow_feats_tensor is not None:
        save_dict['flow_feats'] = flow_feats_tensor
        print(f"[INFO] **************** stage1 flow_feats tensors")

    # 重要：compressed 很慢。训练阶段建议先用 uncompressed。
    # np.savez(save_path, **save_dict)
    np.savez_compressed(save_path, **save_dict)

    print(f"[INFO] saved precomputed stage1 tensors: {save_path}")
    print(f"[INFO] Keys in saved file: {list(save_dict.keys())}")
    print("[INFO] data_io.py ------ generate_and_save_stage1_tensors_fast --- end")

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