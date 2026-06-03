import numpy as np
import torch
from torch.utils.data import Dataset


class PrecomputedFlowDataset(Dataset):
    """Dataset for Stage1 using precomputed tensors.
    Supports all three modes (方案A, 方案B, 方案C) by checking for flow_feats key.
    """

    def __init__(self, npz_path: str):
        print(f"[INFO] Dataset ------ Loading {npz_path}")
        data = np.load(npz_path, allow_pickle=True)

        self.x = data['x']  # [N, max_seq_len, feature_dim]
        self.time = data['time']  # [N, max_seq_len]
        self.mask = data['mask']  # [N, max_seq_len]
        self.labels = data['labels']  # [N]
        self.flow_ids = data['flow_ids']  # [N]

        # # 检查重复
        # unique_ids, counts = np.unique(np.array(self.flow_ids, dtype=np.int64), return_counts=True)
        # dup_ids = unique_ids[counts > 1]
        # print(f"[INFO] ！！！！！！！！！！！ PrecomputedFlowDataset  flow_ids shape: {self.flow_ids.shape}")
        # # 这里也没有重复的
        # if len(dup_ids) > 0:
        #     total = (counts[counts > 1] - 1).sum()  # 重复的总条目数（排除第一次出现）
        #     examples = dup_ids[:10].tolist()
        #     print(
        #         f"[INFO] PrecomputedFlowDataset---------------Total duplicated: {total} Examples: {examples}")

        # Optional Stage2 metadata
        self.flow_start_timestamp_us = (
            data["flow_start_timestamp_us"]
            if "flow_start_timestamp_us" in data
            else None
        )
        self.source_id = (
            data["source_id"]
            if "source_id" in data
            else None
        )
        self.destination_id = (
            data["destination_id"]
            if "destination_id" in data
            else None
        )

        # Check if flow features are present (方案C)
        self.has_flow_feats = 'flow_feats' in data
        if self.has_flow_feats:
            self.flow_feats = data['flow_feats']  # [N, flow_feature_dim]
            print(f"[INFO] Dataset ------ Has separate flow features (方案C)")
            print(f"[INFO]   Flow features shape: {self.flow_feats.shape}")
        else:
            self.flow_feats = None
            print("[INFO] Dataset ------ No separate flow features (方案A or 方案B)")

        self.n_samples = len(self.labels)
        print(f"[INFO] Dataset ------ Loaded {self.n_samples} flows")
        print(f"[INFO]   X shape: {self.x.shape}")
        print(f"[INFO]   Time shape: {self.time.shape}")
        print(f"[INFO]   Mask shape: {self.mask.shape}")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        """
        Returns a dictionary (compatible with existing trainer):
            x: [max_seq_len, feature_dim]
            time: [max_seq_len]
            mask: [max_seq_len]
            label: scalar
            flow_feats: [flow_feature_dim] or None
        """
        item = {
            'x': torch.from_numpy(self.x[idx]).float(),
            'time': torch.from_numpy(self.time[idx]).float(),
            'mask': torch.from_numpy(self.mask[idx]).bool(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'flow_id': torch.tensor(self.flow_ids[idx], dtype=torch.long),
        }

        if self.has_flow_feats:
            item['flow_feats'] = torch.from_numpy(self.flow_feats[idx]).float()
        else:
            item['flow_feats'] = None

        # Stage2 metadata
        if self.flow_start_timestamp_us is not None:
            item["flow_start_timestamp_us"] = torch.tensor(
                int(self.flow_start_timestamp_us[idx]),
                dtype=torch.long,
            )

        # 字符串字段不要转 tensor，保留为 Python str
        if self.source_id is not None:
            item["source_id"] = str(self.source_id[idx])

        if self.destination_id is not None:
            item["destination_id"] = str(self.destination_id[idx])

        return item