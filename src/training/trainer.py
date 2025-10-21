
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

        # Setup callbacks
        callbacks = []
        
        if enable_checkpointing:
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_callback = ModelCheckpoint(
                dirpath='checkpoints',
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

            self.logger = TensorBoardLogger(save_dir='lightning_logs', name=experiment_name)

        # Initialize the Trainer with quieter defaults: no progress bar and
        # no automatic model summary (keeps stdout cleaner). Attach the
        # prepared logger (or False) accordingly.
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
            self._finalize_run(model)
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

    def _finalize_run(self, model: pl.LightningModule):
        """Create a saved_models/<experiment>/version_<N>/ with the last checkpoint and a config JSON.

        This uses the Trainer's logger to discover the current `version` and the
        stored checkpoint callback to find the last/best checkpoint path.
        """
        if self.logger is False or self.checkpoint_callback is False:
            # no logger configured or no checkpoint saved; nothing to do
            return

        try:
            log_dir = self.logger.log_dir

            m = re.search(r"version_(\d+)", log_dir)
            version = m.group(1) if m else '0'

            # create saved_models/<experiment>/version_<N>/ directory
            saved_dir = os.path.join('saved_models', self.experiment_name, f"version_{version}")
            os.makedirs(saved_dir, exist_ok=True)

            # determine checkpoint path
            ckpt_src = self.checkpoint_callback.last_model_path

            # fallback: look for .ckpt files in checkpoints/ directory
            if not os.path.isfile(ckpt_src):
                ckpt_dir = 'checkpoints'
                if os.path.isdir(ckpt_dir):
                    files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
                    if files:
                        # pick the most recent file
                        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                        ckpt_src = files[0]

            # copy checkpoint into saved_dir
            if os.path.isfile(ckpt_src):
                shutil.copy2(ckpt_src, os.path.join(saved_dir, os.path.basename(ckpt_src)))

            # collect config: prefer model.hparams if present
            cfg = {}
            try:
                if hasattr(model, 'hparams') and model.hparams is not None:
                    # hparams can be Namespace or dict-like
                    try:
                        cfg['model_hparams'] = dict(model.hparams)
                    except Exception:
                        # fallback to string representation
                        cfg['model_hparams'] = str(model.hparams)
            except Exception:
                cfg['model_hparams'] = None

            # add training/trainer metadata
            cfg['training'] = {
                'max_epochs': self.max_epochs,
                'accelerator': self.accelerator,
                'log_dir': log_dir,
                'timestamp': __import__('time').ctime()
            }

            # write config.json
            cfg_path = os.path.join(saved_dir, 'config.json')
            with open(cfg_path, 'w', encoding='utf-8') as fh:
                json.dump(cfg, fh, ensure_ascii=False, indent=4)

        except Exception:
            # swallow errors to avoid breaking training pipeline
            import traceback
            traceback.print_exc()