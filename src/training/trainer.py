
# Standard library
from typing import Optional

# For training operations
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
# Fror quieting warnings
import warnings
# For file operations
from pathlib import Path
import shutil
import json

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

        self.checkpoint_callback = enable_checkpointing     # contains false or the callback

        self.enable_early_stopping = enable_early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_mode = early_stopping_mode

        self.logger = enable_logging                        # contains false or the logger
        self.experiment_name = experiment_name

        # Setup callbacks
        callbacks = []
        
        # Resolve project root (two levels above this file: src/training -> repo root)
        self.REPO_ROOT = Path(__file__).parents[2].resolve()
        self.checkpoints_dir = self.REPO_ROOT / 'checkpoints'
        self.logs_dir = self.REPO_ROOT / 'lightning_logs'
        self.saved_models_dir = self.REPO_ROOT / 'saved_models'

        if enable_checkpointing:
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_callback = ModelCheckpoint(
                dirpath=self.checkpoints_dir,
                filename='best-{epoch:02d}-{val_loss:.2f}',
                monitor='val_loss',
                mode='min',
                save_top_k=1,   # the best k models according to the quantity monitored will be saved
                save_last=True
            )
            callbacks.append(checkpoint_callback)
            # keep a reference for later export
            self.checkpoint_callback = checkpoint_callback
        
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
        warnings.filterwarnings("ignore",
            message=r"Checkpoint directory .* exists and is not empty",
        )
        warnings.filterwarnings("ignore",
            message=r"The '.*_dataloader' does not have many workers which may be a bottleneck",
        )

        # Prepare optional logger which allows controlling where lightning
        # writes `lightning_logs/<experiment_name>/version_*`.
        if enable_logging:
            exp_path = self.logs_dir / self.experiment_name
            
            if overwrite_last and exp_path.exists() and exp_path.is_dir():
                # Find all directories starting with "version_"
                versions = []
                for p in exp_path.glob("version_*"):
                    if p.is_dir() and p.name.split("_")[-1].isdigit():
                        versions.append(p)
                
                if versions:
                    # Find the one with the highest version number using lambda function
                    last_version_path = max(versions, key=lambda p: int(p.name.split("_")[-1]))
                    # Remove the directory
                    shutil.rmtree(last_version_path)

            # Initialize the logger
            self.logger = TensorBoardLogger(save_dir=self.logs_dir, name=self.experiment_name)

        # Initialize the Trainer
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            logger=self.logger,
            log_every_n_steps=log_every_n_steps,
            callbacks=callbacks,
            enable_progress_bar=True,
            enable_model_summary=True
        )

    def fit(self, model: pl.LightningModule, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """ Train the model. """
        self.trainer.fit(model, train_loader, val_loader)

        # After training, export last checkpoint and configs to saved_models
        try:
            self.save_model(model)
        except Exception as e:
            # Do not fail training if export fails; print a warning instead
            print(f"WARNING: Could not save model after training: {e}")

    def test(self, model: pl.LightningModule, test_loader: DataLoader):
        """ Test the model. """
        self.trainer.test(model, test_loader)

    def validate(self, model: pl.LightningModule, val_loader: DataLoader):
        """ Validate the model. """
        self.trainer.validate(model, val_loader)

    def save_model(self, model: pl.LightningModule):
        """
        Create a saved_models/<experiment>/version_<N>/ with the last checkpoint and a config JSON.
        """
        if not self.logger or not self.checkpoint_callback:
            # no logger configured or no checkpoint saved; nothing to do
            return

        # Determine current version
        version = getattr(self.logger, 'version', 0)
        
        # Prepare saved model directory
        saved_dir = Path(self.saved_models_dir) / self.experiment_name / f"version_{version}"
        if saved_dir.exists(): shutil.rmtree(saved_dir)
        saved_dir.mkdir(parents=True, exist_ok=True)

        # Copy last Checkpoint
        ckpt_src = Path(self.checkpoint_callback.last_model_path)
        
        if ckpt_src.is_file():
            # Copy to destination keeping the original filename
            shutil.copy2(ckpt_src, saved_dir / ckpt_src.name)

        # Build Configuration
        config = {
            'model_hparams': dict(model.hparams),
            'training': {
                'max_epochs': self.max_epochs,
                'accelerator': self.accelerator,
                'log_dir': str(self.logger.log_dir) 
            }
        }

        # Clean conditional addition
        if self.enable_early_stopping:
            config['training'].update({
                'enable_early_stopping': self.enable_early_stopping,
                'early_stopping_patience': self.early_stopping_patience,
                'early_stopping_monitor': self.early_stopping_monitor,
                'early_stopping_mode': self.early_stopping_mode
            })

        # Write JSON
        with (saved_dir / 'config.json').open('w', encoding='utf-8') as fh:
            json.dump(config, fh, ensure_ascii=False, indent=4)
