
# For model building
import torch
import torch.nn as nn
import pytorch_lightning as pl

class BaseModel(pl.LightningModule):
    """
    Base model class for all neural network models in the project.
    
    This provides a common interface for model saving, loading, and summary.
    """
    
    def __init__(self, criterion: nn.Module, learning_rate: float = 1e-3):
        """ Initialize the base model. """
        super(BaseModel, self).__init__()
        self.learning_rate = learning_rate
        self.criterion = criterion

    def forward(self) -> torch.Tensor:
        """ Forward pass through the model."""
        raise NotImplementedError("Subclasses must implement forward()")
    
    def training_step(self) -> torch.Tensor:
        """ Training step for PyTorch Lightning. """
        raise NotImplementedError("Subclasses must implement training_step()")
    
    def validation_step(self) -> torch.Tensor:
        """ Validation step for PyTorch Lightning. """
        raise NotImplementedError("Subclasses must implement validation_step()")
        
    def test_step(self) -> torch.Tensor:
        """ Test step for PyTorch Lightning. """
        raise NotImplementedError("Subclasses must implement test_step()")
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """ Configure optimizer for PyTorch Lightning. """
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
        criterion: nn.Module,
        dropout: float = 0.5,
        learning_rate: float = 1e-3
    ):

        super(SimpleMLP, self).__init__(criterion, learning_rate)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Save a serializable hyperparameter dict on the model so that training exports (checkpoints/configs) can include the model constructor arguments
        hparams_safe = {
            'class_name': self.__class__.__name__,
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'output_dim': output_dim,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'criterion': criterion.__class__.__name__,
        }
        self.save_hyperparameters(hparams_safe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.network(x)
    
    def training_step(self, batch: tuple) -> torch.Tensor:
        """
        Training step for PyTorch Lightning.
        Args:
            batch: Tuple of (inputs, targets)
        Returns:
            Loss tensor
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: tuple) -> torch.Tensor:
        """
        Validation step for PyTorch Lightning.
        Args:
            batch: Tuple of (inputs, targets)
        Returns:
            Loss tensor
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch: tuple) -> torch.Tensor:
        """
        Test step for PyTorch Lightning.
        Args:
            batch: Tuple of (inputs, targets)
        Returns:
            Loss tensor
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        
        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:

        return torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
