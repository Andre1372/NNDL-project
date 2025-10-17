"""
Training utilities and trainer classes.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class Trainer:
    """
    Trainer class for training PyTorch models.
    Handles the training loop, validation, and model checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,   # Loss function
        optimizer: Optimizer,
        device: str = 'cpu'    
        ):
        """ Initialize the trainer. """

        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

    def train_epoch(self, dataloader: DataLoader, metric_fn: Optional[Callable] = None) -> Dict[str, float]:
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
        
        for sample_batched in dataloader:
            # Move data to device
            inputs = sample_batched[0].to(self.device)
            targets = sample_batched[1].to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            num_batches += 1
            
            if metric_fn is not None:
                metric = metric_fn(outputs, targets)
                running_metric += metric
        
        result = {'loss': running_loss / num_batches}
        if metric_fn is not None:
            result['metric'] = running_metric / num_batches
        
        return result
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader, metric_fn: Optional[Callable] = None) -> Dict[str, float]:
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
        
        for sample_batched in dataloader:
            inputs = sample_batched[0].to(self.device)
            targets = sample_batched[1].to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Update statistics
            running_loss += loss.item()
            num_batches += 1
            
            if metric_fn is not None:
                metric = metric_fn(outputs, targets)
                running_metric += metric
        
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


class LightningTrainer:
    """
    PyTorch Lightning-based trainer wrapper.
    
    This provides a simplified interface to PyTorch Lightning's trainer
    with automatic checkpointing, logging, and best practices.
    """
    
    def __init__(
        self,
        max_epochs: int = 10,
        accelerator: str = 'auto',
        devices: int = 1,
        log_every_n_steps: int = 50,
        enable_checkpointing: bool = True,
        checkpoint_dir: str = 'checkpoints',
        enable_early_stopping: bool = False,
        early_stopping_patience: int = 5,
        early_stopping_monitor: str = 'val_loss',
        early_stopping_mode: str = 'min'
    ):
        """
        Initialize the Lightning trainer.
        
        Args:
            max_epochs: Maximum number of training epochs
            accelerator: Device accelerator ('auto', 'cpu', 'gpu', 'mps')
            devices: Number of devices to use
            log_every_n_steps: Logging frequency
            enable_checkpointing: Whether to save checkpoints
            checkpoint_dir: Directory to save checkpoints
            enable_early_stopping: Whether to use early stopping
            early_stopping_patience: Number of epochs with no improvement after which to stop
            early_stopping_monitor: Metric to monitor for early stopping
            early_stopping_mode: Whether to minimize or maximize the monitored metric
        """
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.devices = devices
        
        # Setup callbacks
        callbacks = []
        
        if enable_checkpointing:
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename='best-{epoch:02d}-{val_loss:.2f}',
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                save_last=True
            )
            callbacks.append(checkpoint_callback)
        
        if enable_early_stopping:
            early_stop_callback = EarlyStopping(
                monitor=early_stopping_monitor,
                patience=early_stopping_patience,
                mode=early_stopping_mode,
                verbose=True
            )
            callbacks.append(early_stop_callback)
        
        # Initialize PyTorch Lightning trainer
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            log_every_n_steps=log_every_n_steps,
            callbacks=callbacks,
            enable_progress_bar=True,
            enable_model_summary=True
        )
    
    def fit(
        self,
        model: pl.LightningModule,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ):
        """
        Train the model.
        
        Args:
            model: PyTorch Lightning module
            train_loader: Training data loader
            val_loader: Optional validation data loader
        """
        self.trainer.fit(model, train_loader, val_loader)
    
    def test(
        self,
        model: pl.LightningModule,
        test_loader: DataLoader
    ):
        """
        Test the model.
        
        Args:
            model: PyTorch Lightning module
            test_loader: Test data loader
        """
        self.trainer.test(model, test_loader)
    
    def validate(
        self,
        model: pl.LightningModule,
        val_loader: DataLoader
    ):
        """
        Validate the model.
        
        Args:
            model: PyTorch Lightning module
            val_loader: Validation data loader
        """
        self.trainer.validate(model, val_loader)
