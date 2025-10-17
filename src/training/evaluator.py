"""
Model evaluation utilities.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional


class Evaluator:
    """
    Evaluator class for testing PyTorch models.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """ Initialize the evaluator. """
        self.model = model.to(device)
        self.device = device
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, criterion: Optional[nn.Module] = None) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: Data loader for evaluation
            criterion: Optional loss function
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        running_loss = 0.0
        num_batches = 0
        
        for sample_batched in dataloader:
            inputs = sample_batched[0].to(self.device)
            targets = sample_batched[1].to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            
            # Calculate loss if criterion provided
            if criterion is not None:
                loss = criterion(outputs, targets)
                running_loss += loss.item()
                num_batches += 1
            
            # Store predictions and targets
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
        
        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        result = {
            'predictions': all_predictions,
            'targets': all_targets
        }
        
        if criterion is not None:
            result['loss'] = running_loss / num_batches
        
        return result
