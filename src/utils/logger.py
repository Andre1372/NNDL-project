"""
Logging utilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = 'nndl',
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


class TensorBoardLogger:
    """
    Simple wrapper for TensorBoard logging.
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory to save TensorBoard logs
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: dict, step: int):
        """Log multiple scalar values."""
        if self.enabled:
            self.writer.add_scalars(tag, values, step)
    
    def log_image(self, tag: str, image, step: int):
        """Log an image."""
        if self.enabled:
            self.writer.add_image(tag, image, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram."""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def close(self):
        """Close the writer."""
        if self.enabled:
            self.writer.close()
