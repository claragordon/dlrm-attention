import numpy as np
import torch
from torch.utils.data import Dataset

class CriteoNPZ(Dataset):
    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.y = d["y"].astype(np.float32)          # [N]
        self.dense = d["dense"].astype(np.float32)  # [N, 13]
        self.sparse = d["sparse"].astype(np.int64)  # [N, 26]

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.dense[idx]),
            torch.from_numpy(self.sparse[idx]),
            torch.tensor(self.y[idx]),
        )