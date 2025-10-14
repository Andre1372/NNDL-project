# Source Code Structure

This directory contains the modular source code for the NNDL (Neural Networks and Deep Learning) project.

## Directory Structure

```
src/
├── __init__.py              # Package initialization
├── hardware_test.py         # Hardware compatibility test script
├── README.md                # This file
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
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── logger.py           # Logging utilities
│   ├── metrics.py          # Evaluation metrics
│   └── visualization.py    # Visualization functions
└── config/                  # Configuration management
    ├── __init__.py
    ├── config.py           # Configuration classes
    └── default_config.json # Default configuration template
```

## Module Descriptions

### 📁 data/
Contains utilities for data loading, preprocessing, and augmentation.
- `dataset.py`: Base dataset classes and data loader creation
- `transforms.py`: Data transformation and augmentation classes

### 📁 models/
Contains PyTorch model definitions and architectures.
- `base_model.py`: Base model class with common functionality (save, load, summary)
- Example models: `SimpleMLP` and `SimpleCNN`

### 📁 training/
Contains training loops and evaluation functions.
- `trainer.py`: Trainer class for model training with progress tracking
- `evaluator.py`: Evaluator class for model testing and prediction

### 📁 utils/
Contains helper functions and utilities.
- `logger.py`: Logging setup and TensorBoard integration
- `metrics.py`: Evaluation metrics (accuracy, precision, recall, F1, etc.)
- `visualization.py`: Plotting functions for training history, confusion matrices, etc.

### 📁 config/
Contains configuration management.
- `config.py`: Configuration dataclasses for experiments
- `default_config.json`: Template configuration file

## Usage Examples

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
