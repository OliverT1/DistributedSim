import torch
import numpy as np

class GPTTrainDataset(torch.utils.data.Dataset):
    """Simple dataset wrapper for training data"""
    def __init__(self, data, device):
        self.examples, self.block_size = data.shape

        self.data = torch.from_numpy(data).to(device=device).long()

    def __len__(self):
        return self.examples

    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1], x[1:]