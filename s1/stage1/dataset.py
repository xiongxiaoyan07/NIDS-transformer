import numpy as np
import torch
from torch.utils.data import Dataset


class PrecomputedFlowDataset(Dataset):
    """
    Dataset for Stage1 using precomputed tensors.
    """
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.x_tensor = torch.from_numpy(data["x"])
        self.time_tensor = torch.from_numpy(data["time"])
        self.mask_tensor = torch.from_numpy(data["mask"])
        self.labels = torch.from_numpy(data["labels"])
        self.flow_ids = torch.from_numpy(data["flow_ids"])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "x": self.x_tensor[index],
            "time": self.time_tensor[index],
            "mask": self.mask_tensor[index],
            "label": self.labels[index],
            "flow_id": self.flow_ids[index],
        }