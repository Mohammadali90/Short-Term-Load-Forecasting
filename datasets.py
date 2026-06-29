"""Shared PyTorch Dataset wrapper used by every sequence model."""

import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """Wraps pre-windowed (X, y) sequence arrays/tensors for use with a
    PyTorch DataLoader."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def to_tensor_dataset(X, y) -> TimeSeriesDataset:
    """Convert numpy arrays to float tensors and wrap them in a
    TimeSeriesDataset."""
    return TimeSeriesDataset(torch.FloatTensor(X), torch.FloatTensor(y))
