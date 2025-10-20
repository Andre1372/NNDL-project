
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional

class BaseModel(pl.LightningModule):
    """
    Base model class for all neural network models in the project.
    
    This provides a common interface for model saving, loading, and summary.
    """
    
    def __init__(self, learning_rate: float = 1e-3, criterion: Optional[nn.Module] = None):
        """ Initialize the base model. """
        super(BaseModel, self).__init__()
        self.learning_rate = learning_rate
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the model."""
        raise NotImplementedError("Subclasses must implement forward()")
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Training step for PyTorch Lightning.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of the batch
            
        Returns:
            Loss tensor
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Validation step for PyTorch Lightning.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of the batch
            
        Returns:
            Loss tensor
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Test step for PyTorch Lightning.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of the batch
            
        Returns:
            Loss tensor
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        
        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizer for PyTorch Lightning.
        
        Returns:
            Optimizer instance
        """
        raise NotImplementedError("Subclasses must implement configure_optimizers()")
    
    def get_num_parameters(self) -> int:
        """ Get the total number of trainable parameters. """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self):
        """Print model summary."""
        print(f"\n{'='*80}")
        print(f"Model: {self.__class__.__name__}")
        print(f"{'='*80}")
        print(self)
        print(f"Total trainable parameters: {self.get_num_parameters():,}")
        print(f"{'='*80}\n")


class SimpleMLP(BaseModel):
    """
    Simple Multi-Layer Perceptron example.
    This is a template/example model architecture.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: list, 
        output_dim: int, 
        dropout: float = 0.5,
        learning_rate: float = 1e-3,
        criterion: Optional[nn.Module] = None
    ):

        super(SimpleMLP, self).__init__(learning_rate=learning_rate, criterion=criterion)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.network(x)
    
    def configure_optimizers(self) -> torch.optim.Optimizer:

        return torch.optim.Adam(self.network.parameters(), lr=1e-2)
