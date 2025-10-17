"""
Training utilities and trainer classes.
"""

from torch.utils.data import DataLoader
from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


class LightningTrainer:
    """
    PyTorch Lightning-based trainer wrapper.
    
    This provides a simplified interface to PyTorch Lightning's trainer
    with automatic checkpointing, logging, and best practices.
    """
    
    def __init__(
        self,
        max_epochs: int = 10,                       # Maximum number of training epochs
        accelerator: str = 'auto',                  # Device accelerator ('auto', 'cpu', 'gpu', 'mps')
        devices: int = 1,                           # Number of devices to use
        log_every_n_steps: int = 50,                # Logging frequency
        enable_checkpointing: bool = True,          # Whether to save checkpoints
        checkpoint_dir: str = 'checkpoints',        # Directory to save checkpoints
        enable_early_stopping: bool = False,        # Whether to use early stopping
        early_stopping_patience: int = 5,           # Number of epochs with no improvement after which to stop
        early_stopping_monitor: str = 'val_loss',   # Metric to monitor for early stopping
        early_stopping_mode: str = 'min'            # Whether to minimize or maximize the monitored metric
    ):
        """ Initialize the Lightning trainer. """

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

    def fit(self, model: pl.LightningModule, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """ Train the model. """
        self.trainer.fit(model, train_loader, val_loader)

    def test(self, model: pl.LightningModule, test_loader: DataLoader):
        """ Test the model. """
        self.trainer.test(model, test_loader)

    def validate(self, model: pl.LightningModule, val_loader: DataLoader):
        """ Validate the model. """
        self.trainer.validate(model, val_loader)
