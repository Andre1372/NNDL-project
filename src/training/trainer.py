"""
Training utilities and trainer classes.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
from tqdm import tqdm


class Trainer:
    """
    Trainer class for training PyTorch models.
    
    Handles the training loop, validation, and model checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: str = 'cpu',
        scheduler: Optional[any] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            criterion: Loss function
            optimizer: Optimizer for training
            device: Device to use ('cpu', 'cuda', or 'mps')
            scheduler: Optional learning rate scheduler
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        metric_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            metric_fn: Optional metric function to compute
            
        Returns:
            Dictionary with 'loss' and optionally 'metric'
        """
        self.model.train()
        
        running_loss = 0.0
        running_metric = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            num_batches += 1
            
            if metric_fn is not None:
                metric = metric_fn(outputs, targets)
                running_metric += metric
                pbar.set_postfix({
                    'loss': running_loss / num_batches,
                    'metric': running_metric / num_batches
                })
            else:
                pbar.set_postfix({'loss': running_loss / num_batches})
        
        result = {'loss': running_loss / num_batches}
        if metric_fn is not None:
            result['metric'] = running_metric / num_batches
        
        return result
    
    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
        metric_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            metric_fn: Optional metric function to compute
            
        Returns:
            Dictionary with 'loss' and optionally 'metric'
        """
        self.model.eval()
        
        running_loss = 0.0
        running_metric = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Validation")
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Update statistics
            running_loss += loss.item()
            num_batches += 1
            
            if metric_fn is not None:
                metric = metric_fn(outputs, targets)
                running_metric += metric
                pbar.set_postfix({
                    'loss': running_loss / num_batches,
                    'metric': running_metric / num_batches
                })
            else:
                pbar.set_postfix({'loss': running_loss / num_batches})
        
        result = {'loss': running_loss / num_batches}
        if metric_fn is not None:
            result['metric'] = running_metric / num_batches
        
        return result
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        metric_fn: Optional[Callable] = None,
        save_best: bool = True,
        checkpoint_path: str = 'best_model.pth'
    ):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            metric_fn: Optional metric function to compute
            save_best: Whether to save the best model
            checkpoint_path: Path to save the best model
        """
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 80)
            
            # Train
            train_result = self.train_epoch(train_loader, metric_fn)
            self.train_losses.append(train_result['loss'])
            
            # Validate
            val_result = self.validate(val_loader, metric_fn)
            self.val_losses.append(val_result['loss'])
            
            # Print epoch summary
            print(f"Train Loss: {train_result['loss']:.4f}")
            print(f"Val Loss: {val_result['loss']:.4f}")
            
            if metric_fn is not None:
                self.train_metrics.append(train_result['metric'])
                self.val_metrics.append(val_result['metric'])
                print(f"Train Metric: {train_result['metric']:.4f}")
                print(f"Val Metric: {val_result['metric']:.4f}")
            
            # Save best model
            if save_best and val_result['loss'] < best_val_loss:
                best_val_loss = val_result['loss']
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Best model saved to {checkpoint_path}")
            
            # Step scheduler if present
            if self.scheduler is not None:
                self.scheduler.step()
