"""
Data transformation and augmentation utilities.
"""

import torch

class ToTensor:
    """
    Convert input to PyTorch tensor.
    """
    
    def __call__(self, sample):
        x, y = sample
        return (torch.Tensor(x).float(),
                torch.Tensor([y]).float())
