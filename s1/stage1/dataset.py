import numpy as np
import torch
from torch.utils.data import Dataset


class PrecomputedFlowDataset(Dataset):
    """Dataset for Stage1 using precomputed tensors.
    Supports all three modes (方案A, 方案B, 方案C) by checking for flow_feats key.
    """

    def __init__(self, npz_path: str):
        print(f"[INFO] Dataset ------ Loading {npz_path}")
        data = np.load(npz_path)

        self.x = data['x']  # [N, max_seq_len, feature_dim]
        self.time = data['time']  # [N, max_seq_len]
        self.mask = data['mask']  # [N, max_seq_len]
        self.labels = data['labels']  # [N]
        self.flow_ids = data['flow_ids']  # [N]

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
        }

        if self.has_flow_feats:
            item['flow_feats'] = torch.from_numpy(self.flow_feats[idx]).float()
        else:
            item['flow_feats'] = None

        return item