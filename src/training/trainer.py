
from torch.utils.data import DataLoader
from typing import Optional
import warnings
import os
import re
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


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
        log_every_n_steps: int = 1,                 # Logging frequency

        enable_checkpointing: bool = True,          # Whether to save checkpoints

        enable_early_stopping: bool = False,        # Whether to use early stopping
        early_stopping_patience: int = 5,           # Number of epochs with no improvement after which to stop
        early_stopping_monitor: str = 'val_loss',   # Metric to monitor for early stopping
        early_stopping_mode: str = 'min',           # Whether to minimize or maximize the monitored metric

        enable_logging: bool = False,               # Whether to enable TensorBoard logging
        experiment_name: str = 'default',           # Experiment name for logging
        overwrite_last: bool = False,               # Whether to overwrite the last version
    ):
        """ Initialize the Lightning trainer. """

        self.max_epochs = max_epochs
        self.accelerator = accelerator
        
        # Setup callbacks
        callbacks = []
        
        if enable_checkpointing:
            checkpoint_callback = ModelCheckpoint(
                dirpath='checkpoints',
                filename='best-{epoch:02d}-{val_loss:.2f}',
                monitor='val_loss',
                mode='min',
                save_top_k=1,   # the best k models according to the quantity monitored will be saved
                save_last=True
            )
            callbacks.append(checkpoint_callback)
        
        # stop training early if no improvement in monitored metric
        if enable_early_stopping:
            early_stop_callback = EarlyStopping(
                monitor=early_stopping_monitor,
                patience=early_stopping_patience,
                mode=early_stopping_mode,
                verbose=True
            )
            callbacks.append(early_stop_callback)
        
        # Quiet some recurring informational/warning messages coming from PyTorch Lightning
        warnings.filterwarnings(
            "ignore",
            message=r"Checkpoint directory .* exists and is not empty",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"The '.*_dataloader' does not have many workers which may be a bottleneck",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"The number of training batches \(.*\) is smaller than the logging interval",
        )

        # Prepare optional logger which allows controlling where lightning
        # writes `lightning_logs/<experiment_name>/version_*`.
        logger = False
        if enable_logging:
            exp_path = os.path.join('lightning_logs', experiment_name)
            if overwrite_last:
                if os.path.isdir(exp_path):
                    # Find existing version_* subdirectories and pick the highest-numbered one.
                    candidates = []
                    for name in os.listdir(exp_path):
                        m = re.match(r"version_(\d+)$", name)
                        if m:
                            candidates.append((int(m.group(1)), name))
                    if candidates:
                        candidates.sort()
                        _, last_name = candidates[-1]
                        to_remove = os.path.join(exp_path, last_name)
                        if os.path.isdir(to_remove):
                            shutil.rmtree(to_remove)

            logger = TensorBoardLogger(save_dir='lightning_logs', name=experiment_name)

        # Initialize the Trainer with quieter defaults: no progress bar and
        # no automatic model summary (keeps stdout cleaner). Attach the
        # prepared logger (or False) accordingly.
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            logger=logger,
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