"""
Dataset classes for loading and managing data.
"""

import torch
from torch.utils.data import Dataset
from typing import Optional, Callable


class BaseDataset(Dataset):
    """
    Base dataset class for the project.
    
    This is a template that should be extended for specific datasets.
    """
    
    def __init__(
        self,
        data_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the data directory
            transform: Optional transform to apply to inputs
            target_transform: Optional transform to apply to targets
        """
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        
        # Load data here
        self.data = []
        self.targets = []
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (data, target)
        """
        sample = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return sample, target


def get_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset
        batch_size: Batch size for data loading
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Dictionary with 'train', 'val', and optionally 'test' dataloaders
    """
    from torch.utils.data import DataLoader
    
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    }
    
    if test_dataset is not None:
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    return dataloaders
