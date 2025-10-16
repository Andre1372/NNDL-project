# Source Code Structure

This directory contains the description source code for the NNDL-project.

## Directory Structure

```
src/
├── __init__.py              # Package initialization for imports
├── hardware_test.py         # Hardware compatibility test script
├── example_usage.py         # Example of code usage
├── README.md                # This file
├── config/                  # Configuration management
|   ├── __init__.py
|   └── config.py           # Configuration classes
├── data/                    # Data loading and preprocessing
│   ├── __init__.py
│   ├── dataset.py          # Dataset classes
│   └── transforms.py       # Data transformations
├── models/                  # Neural network architectures
│   ├── __init__.py
│   └── base_model.py       # Base model classes and examples
├── training/                # Training and evaluation
│   ├── __init__.py
│   ├── trainer.py          # Training loop and utilities
│   └── evaluator.py        # Model evaluation
└── utils/                   # Utility functions
    ├── __init__.py
    ├── logger.py           # Logging utilities
    ├── metrics.py          # Evaluation metrics
    └── visualization.py    # Visualization functions
```

## Configuration Module

The configuration module contains only one file: `config.py`.
This file contains helpful classes and functions to manage project settings through JSON files.

### `config.py`

##### Classes:

- `DataConfig` → Manages data loading and preprocessing settings
- `ModelConfig` → Defines model architecture parameters
- `TrainingConfig` → Handles training hyperparameters
- `ExperimentConfig` → Main configuration class that contains all sub-configurations (`DataConfig`, `ModelConfig`, `TrainingConfig`) and manages experiment-level settings like naming, device selection, and directory paths.

##### Functions:
- `load_config(config_path: str) -> ExperimentConfig`
Loads configuration from a JSON file and creates an `ExperimentConfig` object with all nested configurations.

### Typical usage

1. Start by creating a JSON file like `configs/default_config.json`
2. Then you can load the configurations like in this example
```python
from src.config.config import load_config

config = load_config("configs/default_config.json")

# Access all the parameters
print(f"Experiment: {config.experiment_name}")
print(f"Model type: {config.ModelConfig.model_type}")
```


## Data Module

The data module handles dataset class, preprocessing, and transformations for loading correctly a dataset using `torch.utils.data.DataLoader`. It contains two files `dataset.py`, `transforms.py`.

### `dataset.py`

##### Classes:
- `BaseDataset` → Implement a dataset compatible with torch **!TO COMPLETE!**

### `transforms.py`

##### Classes:
- `ToTensor` → Callable class to convert a data sample to a tensor.

### Typical usage

```python
from src.data.dataset import BaseDataset
from src.data.transforms import ToTensor
from torch.utils.data import DataLoader

# Create transformation (can be concatenated using torchvision.transforms.Compose)
to_tensor = ToTensor()

# Create dataset with transformation
train_dataset = BaseDataset(
    data_path="data/train_data", 
    transform=to_tensor
)

# Access transformed samples samples
sample = train_dataset[0]

# Use a dataloader
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
```


## Models Module

The models module contains neural network architectures. Includes only the file `base_model.py`.

### `base_model.py`

##### Classes:
- `BaseModel` → Abstract base class providing common interface for all models with save/load methods, parameter counting, and model summary
- `SimpleMLP` → Example implementation of Multi-Layer Perceptron
- `SimpleCNN` → Example implementation of CNN

### Typical usage

How to create a new model class:

```python
# in base_model.py
class CustomModel(BaseModel):
    """ Custom neural network architecture. """
    
    def __init__(self, parameters):
        super(CustomModel, self).__init__()
        """ Define the layers. """
    
    def forward(self, x):
        """ Return the NN output given the input. """
```
Once created, the model will be passed to a trainer.



## Training Module

The training module provides classes for training and evaluating NN models. It contains two files `trainer.py`, `evaluator.py`.

### `trainer.py`

##### Classes:
- `Trainer` → Training manager that handles the complete training workflow including training loops, validation, metrics tracking, and model checkpointing with automatic best model saving functionality.

### `evaluator.py`

##### Classes:
- `Evaluator` → Evaluation class for testing models.

### Typical usage

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator

model = # model from base_model.py
criterion = # loss function from torch.nn
optimizer = # optimizer from torch.optim

# Create trainer instance
trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, device='cuda')

# Train the model
trainer.fit(
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    num_epochs=100,
    metric_fn=your_metric_function,  # optional
    save_best=True,
    checkpoint_path='best_model.pth'
)

# Evaluate the model
evaluator = Evaluator(model=model, device='cuda')
results = evaluator.evaluate(
    dataloader=test_dataloader,
    criterion=criterion  # optional
)

```



### Basic Training Pipeline

```python
import torch
from src.config.config import load_config
from src.data.dataset import get_dataloaders, BaseDataset
from src.models.base_model import SimpleMLP
from src.training.trainer import Trainer
from src.utils.logger import setup_logger

# Load configuration
config = load_config('src/config/default_config.json')

# Setup logger
logger = setup_logger('training', log_file='logs/train.log')

# Create datasets
train_dataset = BaseDataset(config.data.data_path, transform=None)
val_dataset = BaseDataset(config.data.data_path, transform=None)

# Create data loaders
dataloaders = get_dataloaders(
    train_dataset,
    val_dataset,
    batch_size=config.data.batch_size,
    num_workers=config.data.num_workers
)

# Create model
model = SimpleMLP(
    input_dim=config.model.input_dim,
    hidden_dims=config.model.hidden_dims,
    output_dim=config.model.output_dim,
    dropout=config.model.dropout
)

# Setup training
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=config.device
)

# Train model
trainer.fit(
    train_loader=dataloaders['train'],
    val_loader=dataloaders['val'],
    num_epochs=config.training.num_epochs,
    save_best=True,
    checkpoint_path='checkpoints/best_model.pth'
)
```

### Model Evaluation

```python
from src.training.evaluator import Evaluator
from src.utils.metrics import accuracy, precision_recall_f1
from src.utils.visualization import plot_confusion_matrix

# Create evaluator
evaluator = Evaluator(model, device=config.device)

# Evaluate on test set
results = evaluator.evaluate(dataloaders['test'], criterion=criterion)

# Compute metrics
acc = accuracy(results['predictions'], results['targets'])
precision, recall, f1 = precision_recall_f1(results['predictions'], results['targets'])

print(f"Test Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
```

### Visualization

```python
from src.utils.visualization import (
    plot_training_history,
    plot_confusion_matrix,
    visualize_predictions
)
from src.utils.metrics import get_confusion_matrix

# Plot training history
plot_training_history(
    train_losses=trainer.train_losses,
    val_losses=trainer.val_losses,
    save_path='results/training_history.png'
)

# Plot confusion matrix
cm = get_confusion_matrix(results['predictions'], results['targets'])
plot_confusion_matrix(cm, class_names=None, save_path='results/confusion_matrix.png')
```

## Best Practices

1. **Configuration Management**: Always use configuration files for experiments
2. **Logging**: Use the logger module for consistent logging across components
3. **Checkpointing**: Save model checkpoints regularly during training
4. **Modular Design**: Keep components loosely coupled and reusable
5. **Type Hints**: Use type hints for better code documentation
6. **Docstrings**: Document all classes and functions with clear docstrings

## Adding New Components

### Adding a New Model

Create a new file in `models/` that inherits from `BaseModel`:

```python
from src.models.base_model import BaseModel
import torch.nn as nn

class MyCustomModel(BaseModel):
    def __init__(self, ...):
        super(MyCustomModel, self).__init__()
        # Define your layers
    
    def forward(self, x):
        # Define forward pass
        return x
```

### Adding a New Dataset

Create a new dataset class in `data/dataset.py` that inherits from `BaseDataset` or PyTorch's `Dataset`:

```python
class MyCustomDataset(BaseDataset):
    def __init__(self, data_path, transform=None):
        super(MyCustomDataset, self).__init__(data_path, transform)
        # Load your data
    
    def __getitem__(self, idx):
        # Return sample and target
        return sample, target
```

### Adding New Metrics

Add new metric functions in `utils/metrics.py`:

```python
def my_custom_metric(predictions, targets):
    # Compute your metric
    return metric_value
```

## Testing Hardware

To verify your PyTorch installation and hardware compatibility, run:

```bash
python src/hardware_test.py
```

This will test your GPU/CPU and provide performance benchmarks.
