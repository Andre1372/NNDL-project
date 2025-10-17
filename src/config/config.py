"""
Configuration management for the project.
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class DataConfig:
    """
    Configuration for data loading and preprocessing.
    """
    data_path: str
    batch_size: int
    num_workers: int    # Number of processes used for data loading
    train_split: float
    val_split: float
    test_split: float
    shuffle: bool
    pin_memory: bool    # Pin memory for faster data transfer to GPU


@dataclass
class ModelConfig:
    """
    Configuration for model architecture.
    """
    model_type: str
    input_dim: int
    hidden_dims: list
    output_dim: int
    dropout: float


@dataclass
class TrainingConfig:
    """
    Configuration for training.
    """
    num_epochs: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    loss_function: str
    early_stopping_patience: int
    gradient_clip: Optional[float] # can be null


@dataclass
class ExperimentConfig:
    """
    Main configuration combining all sub-configs.
    """
    experiment_name: str
    seed: int
    device: str     # 'cpu', 'cuda', or 'mps'
    checkpoint_dir: str
    log_dir: str
    save_frequency: int
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def load_config(config_path: str) -> ExperimentConfig:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to configuration file (.json)
        
    Returns:
        ExperimentConfig object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load configuration from file
    with open(config_path, 'r') as f:
        if config_path.suffix == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}. Only .json is supported.")
    
    # Create configuration objects
    data_config = DataConfig(**config_dict.get('data', {}))
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    
    # Obtain config for main ExperimentConfig
    config_dict_main = {k: v for k, v in config_dict.items() 
                        if k not in ['data', 'model', 'training']}
    
    # Create main config
    experiment_config = ExperimentConfig(
        **config_dict_main,
        data=data_config,
        model=model_config,
        training=training_config
    )
    
    return experiment_config
