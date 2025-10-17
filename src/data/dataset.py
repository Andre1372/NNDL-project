"""
Dataset classes for loading and managing data.
"""

import torch
from torch.utils.data import Dataset
from typing import Optional, Callable


class BaseDataset(Dataset):
    """
    Base dataset class for the project.
    """

    def __init__(self, data_path: str, transform: Optional[Callable] = None):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the data directory
            transform: Optional transform to apply to inputs
        """
        self.data_path = data_path
        self.transform = transform
        
        # Load data here using data_path
        # TODO!!
        self.data = []
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int):
        """Get a sample by index."""
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
