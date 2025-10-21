
from torch.utils.data import DataLoader
from typing import Optional
import warnings
import os
import re
import shutil
import json
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
        self.checkpoint_callback = enable_checkpointing     # contains false or the callback
        self.logger = enable_logging                        # contains false or the logger
        self.experiment_name = experiment_name
        self.enable_early_stopping = enable_early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_mode = early_stopping_mode

        # Setup callbacks
        callbacks = []
        
        # Resolve project root (two levels above this file: src/training -> repo root)
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.checkpoints_dir = os.path.join(self.repo_root, 'checkpoints')
        self.logs_dir = os.path.join(self.repo_root, 'lightning_logs')
        self.saved_models_dir = os.path.join(self.repo_root, 'saved_models')

        if enable_checkpointing:
            os.makedirs(self.checkpoints_dir, exist_ok=True)
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
            exp_path = os.path.join(self.logs_dir, experiment_name)
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

            # Use absolute logs_dir so Lightning writes logs under the repository root
            self.logger = TensorBoardLogger(save_dir=self.logs_dir, name=experiment_name)

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
        except Exception:
            # Do not fail training if export fails; print a warning instead
            import traceback
            traceback.print_exc()

    def test(self, model: pl.LightningModule, test_loader: DataLoader):
        """ Test the model. """
        self.trainer.test(model, test_loader)

    def validate(self, model: pl.LightningModule, val_loader: DataLoader):
        """ Validate the model. """
        self.trainer.validate(model, val_loader)

    def save_model(self, model: pl.LightningModule):
        """Create a saved_models/<experiment>/version_<N>/ with the last checkpoint and a config JSON.

        This uses the Trainer's logger to discover the current `version` and the
        stored checkpoint callback to find the last/best checkpoint path.
        """
        if self.logger is False or self.checkpoint_callback is False:
            # no logger configured or no checkpoint saved; nothing to do
            return

        log_dir = self.logger.log_dir

        m = re.search(r"version_(\d+)", log_dir)
        version = m.group(1) if m else '0'

        # create saved_models/<experiment>/version_<N>/ directory under repo root
        saved_dir = os.path.join(self.saved_models_dir, self.experiment_name, f"version_{version}")
        os.makedirs(saved_dir, exist_ok=True)

        # determine checkpoint path
        ckpt_src = self.checkpoint_callback.last_model_path

        # copy checkpoint into saved_dir
        if os.path.isfile(ckpt_src):
            shutil.copy2(ckpt_src, os.path.join(saved_dir, os.path.basename(ckpt_src)))

        # collect config: prefer model.hparams if present
        cfg = {}
        cfg['model_hparams'] = dict(model.hparams)

        # add training/trainer metadata
        cfg['training'] = {
            'max_epochs': self.max_epochs,
            'accelerator': self.accelerator,
            'log_dir': log_dir
        }
        if self.enable_early_stopping:
            cfg['training']['enable_early_stopping'] = self.enable_early_stopping
            cfg['training']['early_stopping_patience'] = self.early_stopping_patience
            cfg['training']['early_stopping_monitor'] = self.early_stopping_monitor
            cfg['training']['early_stopping_mode'] = self.early_stopping_mode

        # write config.json
        cfg_path = os.path.join(saved_dir, 'config.json')
        with open(cfg_path, 'w', encoding='utf-8') as fh:
            json.dump(cfg, fh, ensure_ascii=False, indent=4)
