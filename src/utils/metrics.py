
import torch
from typing import Tuple
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import numpy as np


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """ Calculate classification accuracy. """

    pred_classes = predictions.argmax(dim=1) # indices of the maximum value in the tensor
    correct = (pred_classes == targets).sum().item()
    total = targets.size(0)
    return correct / total


def top_k_accuracy(predictions: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float: 
    """ Calculate top-k accuracy. """

    _, top_k_preds = predictions.topk(k, dim=1, largest=True, sorted=True) # indices of top k predictions
    targets_expanded = targets.view(-1, 1).expand_as(top_k_preds)
    correct = (top_k_preds == targets_expanded).any(dim=1).sum().item()
    total = targets.size(0)
    return correct / total


def precision_recall_f1(predictions: torch.Tensor, targets: torch.Tensor, average: str = 'macro') -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth labels
        average: Averaging method ('micro', 'macro', 'weighted')
        
    Returns:
        Tuple of (precision, recall, f1_score)
    """
    pred_classes = predictions.argmax(dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    precision = precision_score(targets_np, pred_classes, average=average, zero_division=0)
    recall = recall_score(targets_np, pred_classes, average=average, zero_division=0)
    f1 = f1_score(targets_np, pred_classes, average=average, zero_division=0)
    
    return precision, recall, f1


def get_confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray: 
    """ Calculate confusion matrix. """

    pred_classes = predictions.argmax(dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    return confusion_matrix(targets_np, pred_classes)


class AverageMeter:
    """ Computes and stores the average and current value. """
    
    def __init__(self):
        """Initialize the meter."""
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update the meter with a new value.
        
        Args:
            val: New value to add
            n: Number of samples the value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricTracker:
    """ Track multiple metrics during training. """
    
    def __init__(self, *metrics: str):
        """
        Initialize tracker with metric names.
        
        Args:
            *metrics: Names of metrics to track
        """
        self.metrics = {metric: AverageMeter() for metric in metrics} # dict of metric name to AverageMeter
    
    def update(self, metric_dict: dict, n: int = 1):
        """
        Update all metrics.
        
        Args:
            metric_dict: Dictionary of metric names and values
            n: Number of samples
        """
        for metric, value in metric_dict.items():
            if metric in self.metrics:
                self.metrics[metric].update(value, n)
    
    def reset(self):
        """ Reset all metrics. """
        for metric in self.metrics.values():
            metric.reset()
    
    def get_averages(self) -> dict:
        """
        Get average values for all metrics.
        
        Returns:
            Dictionary of metric names and average values
        """
        return {name: meter.avg for name, meter in self.metrics.items()}
