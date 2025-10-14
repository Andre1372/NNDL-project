"""
Configuration management for the project.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    data_path: str = './data'
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    shuffle: bool = True
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    model_type: str = 'simple_mlp'
    input_dim: int = 784
    hidden_dims: list = None
    output_dim: int = 10
    dropout: float = 0.5
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


@dataclass
class TrainingConfig:
    """Configuration for training."""
    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    optimizer: str = 'adam'
    scheduler: Optional[str] = None
    scheduler_params: Optional[dict] = None
    loss_function: str = 'cross_entropy'
    early_stopping_patience: int = 5
    gradient_clip: Optional[float] = None
    
    def __post_init__(self):
        if self.scheduler_params is None:
            self.scheduler_params = {}


@dataclass
class ExperimentConfig:
    """Main configuration combining all sub-configs."""
    experiment_name: str = 'default_experiment'
    seed: int = 42
    device: str = 'cpu'  # 'cpu', 'cuda', or 'mps'
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    save_frequency: int = 1
    
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()


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
    
    # Create nested dataclass objects
    data_config = DataConfig(**config_dict.get('data', {}))
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    
    # Remove nested configs from main dict
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


def save_config(config: ExperimentConfig, save_path: str):
    """
    Save configuration to a JSON file.
    
    Args:
        config: ExperimentConfig object to save
        save_path: Path to save the configuration (.json)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dictionary
    config_dict = {
        'experiment_name': config.experiment_name,
        'seed': config.seed,
        'device': config.device,
        'checkpoint_dir': config.checkpoint_dir,
        'log_dir': config.log_dir,
        'save_frequency': config.save_frequency,
        'data': asdict(config.data),
        'model': asdict(config.model),
        'training': asdict(config.training)
    }
    
    # Save to file
    with open(save_path, 'w') as f:
        if save_path.suffix == '.json':
            json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {save_path.suffix}. Only .json is supported.")
    
    print(f"Configuration saved to {save_path}")


def get_default_config() -> ExperimentConfig:
    """
    Get default configuration.
    
    Returns:
        Default ExperimentConfig object
    """
    return ExperimentConfig()
