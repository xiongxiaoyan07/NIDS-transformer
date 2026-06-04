from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class Stage2Dataset(Dataset):
    def __init__(
        self,
        meta_df_sorted: pd.DataFrame,
        z_sorted: np.ndarray,
        context_indices: List[np.ndarray],
        target_split: str,
    ):
        # meta_df_sorted 每条 flow 的元信息，例如 split、label、flow_id
        # reset_index 重置索引，保证 dataframe 的行号---->z_sorted[idx]、context_indices[idx]、meta_df.iloc[idx] 能严格对应
        self.meta_df = meta_df_sorted.reset_index(drop=True)
        # z_sorted 每条 flow 对应的 Stage1 embedding
        self.z = z_sorted.astype(np.float32)
        # context_indices 每条 flow 的上下文 flow 索引
        # self.context_indices[3] = np.array([1, 3]) 表示第3条flow的上下文是第1条和第3条flow。
        self.context_indices = context_indices
        # target_split 当前 Dataset 用 train / val / test 哪个 split
        self.target_split = target_split
        # 找出属于目标 split 的所有行号。
        self.target_rows = self.meta_df.index[self.meta_df["split"] == target_split].to_numpy(dtype=np.int64)

        if len(self.target_rows) == 0:
            raise ValueError(f"No rows found for split={target_split}")
        # 提前取出当前 split 的标签。
        self.labels = self.meta_df.loc[self.target_rows, "label"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.target_rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        row_idx = int(self.target_rows[i])
        ctx_idx = self.context_indices[row_idx]
        # 根据上下文索引取 embedding。
        if len(ctx_idx) == 0:
            context_z = np.zeros((0, self.z.shape[1]), dtype=np.float32)
        else:
            context_z = self.z[ctx_idx]

        return {
            "context_z": torch.from_numpy(context_z).float(),
            "mask": torch.ones(len(ctx_idx), dtype=torch.bool),
            # "label": torch.tensor(int(self.meta_df.at[row_idx, "label"]), dtype=torch.long),
            "label": torch.tensor(int(self.labels[i]), dtype=torch.long),
            "flow_id": int(self.meta_df.at[row_idx, "flow_id"]),
            "row_idx": row_idx,
        }

def stage2_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    # embedding 维度。
    d_model = batch[0]["context_z"].shape[1]
    # 找当前 batch 中最长的上下文长度。
    max_len = max(1, max(int(item["context_z"].shape[0]) for item in batch))

    x = torch.zeros(batch_size, max_len, d_model, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    labels = torch.stack([item["label"] for item in batch])
    flow_ids = torch.tensor([item["flow_id"] for item in batch], dtype=torch.long)
    row_idx = torch.tensor([item["row_idx"] for item in batch], dtype=torch.long)

    for i, item in enumerate(batch):
        length = int(item["context_z"].shape[0])
        if length > 0:
            # 这是右填充
            # x[i, :length] = item["context_z"]
            # mask[i, :length] = True
            # 这是左填充
            # 左填充（leftpadding）意味着：
            # 真实token对齐在右边
            # padding填充在左边
            # mask对应位置要标记真实token
            x[i, max_len - length:] = item["context_z"]
            mask[i, max_len - length:] = True

    return {
        "context_z": x,
        "mask": mask,
        "label": labels,
        "flow_id": flow_ids,
        "row_idx": row_idx,
    }