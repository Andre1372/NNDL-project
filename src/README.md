# Source Code Structure

This directory contains the description source code for the NNDL-project.

## Directory Structure

```
src/
├── __init__.py              # Package initialization for imports
├── hardware_test.py         # Hardware compatibility test script
├── config/                  # Configuration management
|   ├── __init__.py
|   └── config.py           # Save and load configurations
├── data/                       # Data loading and preprocessing
│   ├── __init__.py
│   ├── dataset.py          # Dataset class
│   └── transforms.py       # Data transformations
├── models/                     # Neural network architectures
│   ├── __init__.py
│   └── base_model.py       # Base model class and example
├── training/                   # Training and evaluation
│   ├── __init__.py
|   ├── metrics.py          # Evaluation metrics
│   └── trainer.py          # Training loop and utilities
└── utils/                      # Utility functions
    ├── __init__.py
    └── visualization.py    # Visualization functions
```

## Configuration Module

The configuration module contains only one file: `config.py`.
This file contains helpful classes and functions to manage project settings through JSON files.

### `config.py`



## Data Module

The data module handles dataset class, preprocessing, and transformations for loading correctly a dataset using `torch.utils.data.DataLoader`. It contains two files `dataset.py`, `transforms.py`.

### `dataset.py`

##### Classes:
- `PolyDataset` → Implement a dataset for the example_usage file **!We will need a new dataset class!**

### `transforms.py`

##### Classes:
- `ToTensor` → Callable class to convert a data sample to a tensor.



## Models Module

The models module contains neural network architectures. Includes only the file `base_model.py`.

### `base_model.py`

##### Classes:
- `BaseModel` → Abstract base class providing common interface for all models with save/load methods, parameter counting, and model summary. All successive models should extend this class.
- `SimpleMLP` → Example implementation of Multi-Layer Perceptron, extends BaseModel.

When extending BaseModel we should (see SimpleMLP):
- execute the parent constructor
- save the hyper parameters for logging
- overwrite `forward()`
- overwrite`configure_optimizers()`



## Training Module

The training module provides classes for training and evaluating NN models. It contains two files `trainer.py`, `metrics.py`.

### `trainer.py`

##### Classes:
- `LightningTrainer` → Training wrapper that provides a simplified interface to PyTorch Lightning's trainer with automatic checkpointing, logging, callbacks, and best practices.

### `metrics.py`

File to be filled out which should contains useful metrics to log inside `training_step()`, `validation_step()` of `BaseModel()`.



## Utils Module

The utils module should provide utility functions and classes for visualization and maby others. Until now it contains one file `visualization.py`.

### `visualization.py`

Empty, to be filled out.
