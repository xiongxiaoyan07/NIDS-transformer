# dataset.py
from __future__ import annotations

from typing import Dict, List, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import Stage1Preprocessor

class Stage1FlowDataset(Dataset):
    """
    Each item is one flow:
        x: [max_seq_len, input_dim]
        time: [max_seq_len]
        mask: [max_seq_len]
        y: scalar label
    """
    def __init__(
        self,
        packets: pd.DataFrame,
        flows: pd.DataFrame,
        flow_ids: List[int],
        preprocessor: Stage1Preprocessor,
        cfg: Dict[str, Any],
    ):
        self.packets = packets.copy()
        self.flows = flows.copy()
        self.flow_ids = list(flow_ids)
        self.preprocessor = preprocessor
        self.cfg = cfg

        data_cfg = cfg.get("data", {})
        seq_cfg = cfg.get("sequence", {})

        self.flow_id_col = data_cfg.get("flow_id_col", "flow_id")
        self.label_col = data_cfg.get("label_col", "label")
        self.packet_time_col = data_cfg.get("packet_time_col", "timestamp_us")
        self.packet_iat_col = data_cfg.get("packet_iat_col", "flow_iat_us")

        self.max_seq_len = int(seq_cfg.get("max_seq_len", 64))

        # 将 packet 在 read_stage1_csvs 已经截取好了，这里不需要再分组
        self.flow_rows = {int(row[self.flow_id_col]): row for _, row in self.flows.iterrows()}

    def __len__(self):
        return len(self.flow_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        print("-----------dataset.py-------------------getitem", index)
        flow_id = int(self.flow_ids[index])
        if flow_id not in self.flow_rows:
            raise KeyError(f"flow_id={flow_id} not found in flows.")

        # 获取该 flow 的 packet
        pkt_df = self.packets[self.packets["flow_id"] == flow_id].sort_values(self.packet_time_col)
        flow_df = pd.DataFrame([self.flow_rows[flow_id]])

        packet_x = self.preprocessor.transform_packets(pkt_df)
        flow_x = self.preprocessor.transform_flows(flow_df)

        # Repeat flow features for each packet
        flow_x_tiled = np.repeat(flow_x, repeats=len(pkt_df), axis=0)

        # x_i,t = [packet feature h_i,t and tau_i,t ; flow statistics s_i]
        x = np.concatenate([packet_x, flow_x_tiled], axis=1).astype(np.float32)

        # Raw time for time-aware encoding.
        if self.packet_iat_col in pkt_df.columns:
            time_raw = pd.to_numeric(
                pkt_df[self.packet_iat_col], errors="coerce"
            ).fillna(0).clip(lower=0).to_numpy(dtype=np.float32)
        else:
            time_raw = np.zeros(len(pkt_df), dtype=np.float32)

        # Log scale is more stable for large inter-arrival times.
        time_log = np.log1p(time_raw).astype(np.float32)

        real_len = x.shape[0]
        feature_dim = x.shape[1]

        padded_x = np.zeros((self.max_seq_len, feature_dim), dtype=np.float32)
        padded_time = np.zeros((self.max_seq_len,), dtype=np.float32)
        mask = np.zeros((self.max_seq_len,), dtype=bool)

        padded_x[:real_len] = x
        padded_time[:real_len] = time_log
        mask[:real_len] = True

        label = int(self.flow_rows[flow_id][self.label_col])

        return {
            "x": torch.from_numpy(padded_x),
            "time": torch.from_numpy(padded_time),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(label, dtype=torch.long),
            "flow_id": torch.tensor(flow_id, dtype=torch.long),
        }