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

The models module contains neural network architectures using PyTorch Lightning. Includes only the file `base_model.py`.

### `base_model.py`

##### Classes:
- `BaseModel` → Abstract base class providing common interface for all models with save/load methods, parameter counting, and model summary. Now inherits from `pl.LightningModule` for PyTorch Lightning support.
- `SimpleMLP` → Example implementation of Multi-Layer Perceptron with Lightning support
- `SimpleCNN` → Example implementation of CNN with Lightning support

##### PyTorch Lightning Methods:
All models now include:
- `training_step()` → Defines training logic for one batch
- `validation_step()` → Defines validation logic for one batch
- `test_step()` → Defines testing logic for one batch
- `configure_optimizers()` → Configures optimizer(s) for training

### Typical usage

How to create a new model class:

```python
# in base_model.py
import pytorch_lightning as pl

class CustomModel(BaseModel):
    """ Custom neural network architecture. """
    
    def __init__(self, parameters, learning_rate=1e-3, criterion=None):
        super(CustomModel, self).__init__(learning_rate=learning_rate, criterion=criterion)
        """ Define the layers. """
    
    def forward(self, x):
        """ Return the NN output given the input. """
```
Once created, the model will be passed to a PyTorch Lightning trainer.



## Training Module

The training module provides classes for training and evaluating NN models using PyTorch Lightning. It contains two files `trainer.py`, `evaluator.py`.

### `trainer.py`

##### Classes:
- `Trainer` → (Legacy) Traditional training manager that handles the complete training workflow including training loops, validation, metrics tracking, and model checkpointing with automatic best model saving functionality.
- `LightningTrainer` → **New PyTorch Lightning-based trainer wrapper** that provides a simplified interface to PyTorch Lightning's trainer with automatic checkpointing, logging, callbacks, and best practices. **This is the recommended approach.**

##### PyTorch Lightning Benefits:
- Automatic device placement (CPU/GPU/TPU)
- Built-in callbacks (ModelCheckpoint, EarlyStopping, etc.)
- Automatic logging to TensorBoard
- Progress bars and training monitoring
- Multi-GPU support
- Less boilerplate code

### `evaluator.py`

##### Classes:
- `Evaluator` → Evaluation class for testing models.

### Typical usage

**Using PyTorch Lightning (Recommended):**

```python
import torch.nn as nn
from torch.utils.data import DataLoader
from src.training.trainer import LightningTrainer
from src.models.base_model import SimpleMLP

# Create model (automatically a LightningModule)
model = SimpleMLP(
    input_dim=784,
    hidden_dims=[256, 128],
    output_dim=10,
    learning_rate=1e-3,
    criterion=nn.CrossEntropyLoss()
)

# Create Lightning trainer with desired configuration
trainer = LightningTrainer(
    max_epochs=100,
    accelerator='auto',  # automatically selects GPU/CPU
    enable_checkpointing=True,
    checkpoint_dir='checkpoints',
    enable_early_stopping=True,
    early_stopping_patience=5
)

# Train the model
trainer.fit(
    model=model,
    train_loader=train_dataloader,
    val_loader=val_dataloader
)

# Test the model
trainer.test(model=model, test_loader=test_dataloader)
```

**Using Legacy Trainer:**

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


## Utils Module

The utils module provides utility functions and classes for model evaluation and result visualization. It contains two files `metrics.py` and `visualization.py`.

### `metrics.py`

##### Classes:
- `AverageMeter` → Utility class that computes and stores the average and current value of metrics, useful for tracking running averages during training
- `MetricTracker` → Advanced metric tracking class that manages multiple AverageMeter instances for comprehensive training monitoring

##### Functions:
- `accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float` → Calculate classification accuracy by comparing predicted classes with ground truth labels
- `top_k_accuracy(predictions: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float` → Calculate top-k accuracy, checking if the true label is among the k highest predicted probabilities  
- `precision_recall_f1(predictions: torch.Tensor, targets: torch.Tensor, average: str = 'macro') -> Tuple[float, float, float]` → Calculate precision, recall, and F1 score with different averaging methods ('micro', 'macro', 'weighted')
- `get_confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray` → Generate confusion matrix for classification evaluation

### `visualization.py`

##### Functions:
- `plot_training_history(train_losses: List[float], val_losses: List[float], train_metrics: Optional[List[float]] = None, val_metrics: Optional[List[float]] = None, metric_name: str = 'Metric', save_path: Optional[str] = None)` → Create comprehensive plots of training and validation losses and metrics over epochs with optional saving functionality
- `plot_confusion_matrix(cm: np.ndarray, class_names: Optional[List[str]] = None, normalize: bool = False, save_path: Optional[str] = None)` → Generate professional confusion matrix visualizations with optional normalization, class names, and saving capabilities

### Typical usage

```python
import torch
from src.utils.metrics import accuracy, MetricTracker, get_confusion_matrix
from src.utils.visualization import plot_training_history, plot_confusion_matrix

# Calculate accuracy
acc = accuracy(predictions, targets)
print(f"Accuracy: {acc:.4f}")

# Track metrics during training
tracker = MetricTracker('loss', 'accuracy', 'f1_score')
tracker.update({'loss': 0.5, 'accuracy': 0.85, 'f1_score': 0.82})
averages = tracker.get_averages()

# Generate confusion matrix and visualize
cm = get_confusion_matrix(predictions, targets)
plot_confusion_matrix(
    cm, 
    class_names=[f'Class_{i}' for i in range(10)],
    normalize=True,
    save_path='confusion_matrix.png'
)

# Visualize training progress
plot_training_history(
    train_losses=train_losses,
    val_losses=val_losses, 
    train_metrics=train_acc,
    val_metrics=val_acc,
    metric_name='Accuracy',
    save_path='training_history.png'
)
```

