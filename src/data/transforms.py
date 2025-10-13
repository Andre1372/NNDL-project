"""
Data transformation and augmentation utilities.
"""

import torch
import torch.nn as nn
from typing import List, Callable


class Compose:
    """
    Compose multiple transforms together.
    """
    
    def __init__(self, transforms: List[Callable]):
        """
        Initialize the composition.
        
        Args:
            transforms: List of transforms to apply sequentially
        """
        self.transforms = transforms
    
    def __call__(self, x):
        """Apply all transforms sequentially."""
        for transform in self.transforms:
            x = transform(x)
        return x


class Normalize:
    """
    Normalize tensor with mean and standard deviation.
    """
    
    def __init__(self, mean: float, std: float):
        """
        Initialize normalization.
        
        Args:
            mean: Mean for normalization
            std: Standard deviation for normalization
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization."""
        return (x - self.mean) / self.std


class ToTensor:
    """
    Convert input to PyTorch tensor.
    """
    
    def __call__(self, x):
        """Convert to tensor."""
        if not isinstance(x, torch.Tensor):
            return torch.tensor(x)
        return x


def get_default_transforms(normalize: bool = True):
    """
    Get default data transformations.
    
    Args:
        normalize: Whether to include normalization
        
    Returns:
        Composed transform
    """
    transforms = [ToTensor()]
    
    if normalize:
        transforms.append(Normalize(mean=0.0, std=1.0))
    
    return Compose(transforms)
