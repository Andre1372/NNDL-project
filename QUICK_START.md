# Quick Start Guide

This guide provides quick instructions to start using the NNDL project structure.

## 🚀 Getting Started

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/Andre1372/NNDL-project
cd NNDL-project

# Create virtual environment
python -m venv venv_deep

# Activate (Windows)
venv_deep\Scripts\activate

# Activate (Mac/Linux)
source venv_deep/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Test Installation

```bash
# Test hardware and PyTorch installation
python src/hardware_test.py
```

### 3. Run Example

```bash
# Run the example usage script to see the structure in action
python example_usage.py
```

## 📁 Project Structure Overview

```
src/
├── data/          # Data loading and preprocessing
├── models/        # Model architectures (SimpleMLP, SimpleCNN, etc.)
├── training/      # Trainer and Evaluator classes
├── utils/         # Logging, metrics, visualization
└── config/        # Configuration management
```

## 🔧 Common Tasks

### Create a New Model

```python
from src.models.base_model import BaseModel
import torch.nn as nn

class MyModel(BaseModel):
    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)
```

### Load Configuration

```python
from src.config.config import load_config

# Load from YAML file
config = load_config('src/config/default_config.yaml')

# Or get default config
from src.config.config import get_default_config
config = get_default_config()
```

### Train a Model

```python
from src.training.trainer import Trainer
import torch.nn as nn
import torch.optim as optim

trainer = Trainer(
    model=your_model,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(your_model.parameters(), lr=0.001),
    device='cuda'  # or 'cpu' or 'mps'
)

trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=10,
    save_best=True,
    checkpoint_path='checkpoints/best_model.pth'
)
```

### Evaluate a Model

```python
from src.training.evaluator import Evaluator
from src.utils.metrics import accuracy, precision_recall_f1

evaluator = Evaluator(your_model, device='cuda')
results = evaluator.evaluate(test_loader, criterion=criterion)

acc = accuracy(results['predictions'], results['targets'])
print(f"Accuracy: {acc:.4f}")
```

### Visualize Results

```python
from src.utils.visualization import (
    plot_training_history,
    plot_confusion_matrix
)

# Plot training history
plot_training_history(
    train_losses=trainer.train_losses,
    val_losses=trainer.val_losses,
    save_path='results/history.png'
)

# Plot confusion matrix
from src.utils.metrics import get_confusion_matrix
cm = get_confusion_matrix(predictions, targets)
plot_confusion_matrix(cm, save_path='results/cm.png')
```

## 📊 Using Configuration Files

Create a custom config file `my_experiment.yaml`:

```yaml
experiment_name: my_experiment
device: cuda
seed: 42

data:
  batch_size: 64
  num_workers: 4

model:
  model_type: simple_mlp
  input_dim: 784
  hidden_dims: [512, 256, 128]
  output_dim: 10

training:
  num_epochs: 20
  learning_rate: 0.0001
  optimizer: adam
```

Then load it:

```python
from src.config.config import load_config
config = load_config('my_experiment.yaml')
```

## 🔍 Logging

```python
from src.utils.logger import setup_logger, TensorBoardLogger

# Setup basic logger
logger = setup_logger('training', log_file='logs/train.log')
logger.info('Starting training...')

# Setup TensorBoard logger
tb_logger = TensorBoardLogger('runs/experiment1')
tb_logger.log_scalar('loss', loss_value, step=epoch)
```

## 📖 More Information

- See [src/README.md](src/README.md) for detailed documentation
- Run `python example_usage.py` for a complete working example
- Check individual module files for more examples and usage patterns

## 🆘 Common Issues

**Issue**: `ModuleNotFoundError: No module named 'torch'`
- **Solution**: Make sure PyTorch is installed: `pip install torch`

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size in configuration or use CPU

**Issue**: Import errors
- **Solution**: Make sure you're in the project root and run `python -c "import src"`

## 💡 Best Practices

1. **Always use configuration files** for reproducible experiments
2. **Log everything** - use the logging utilities
3. **Save checkpoints regularly** during training
4. **Visualize results** to understand model behavior
5. **Use version control** (git) for your experiments
6. **Document your changes** and experiments

## 📞 Getting Help

- Read the detailed documentation in `src/README.md`
- Check example implementations in the `src/` modules
- Review `example_usage.py` for working code
