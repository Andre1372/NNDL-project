"""
Model evaluation utilities.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm


class Evaluator:
    """
    Evaluator class for testing PyTorch models.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize the evaluator.
        
        Args:
            model: PyTorch model to evaluate
            device: Device to use ('cpu', 'cuda', or 'mps')
        """
        self.model = model.to(device)
        self.device = device
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        criterion: Optional[nn.Module] = None
    ) -> Dict[str, float]:
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
        
        pbar = tqdm(dataloader, desc="Evaluating")
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Calculate loss if criterion provided
            if criterion is not None:
                loss = criterion(outputs, targets)
                running_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': running_loss / num_batches})
            
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
    
    @torch.no_grad()
    def predict(
        self,
        dataloader: DataLoader
    ) -> torch.Tensor:
        """
        Generate predictions for a dataset.
        
        Args:
            dataloader: Data loader for prediction
            
        Returns:
            Tensor of predictions
        """
        self.model.eval()
        
        all_predictions = []
        
        pbar = tqdm(dataloader, desc="Predicting")
        for inputs, _ in pbar:
            inputs = inputs.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            all_predictions.append(outputs.cpu())
        
        return torch.cat(all_predictions, dim=0)


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    pred_classes = predictions.argmax(dim=1)
    correct = (pred_classes == targets).sum().item()
    total = targets.size(0)
    return correct / total
