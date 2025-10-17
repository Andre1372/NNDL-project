"""
Base model classes and architectures.
"""

import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """
    Base model class for all neural network models in the project.
    
    This provides a common interface for model saving, loading, and summary.
    """
    
    def __init__(self):
        """ Initialize the base model. """
        super(BaseModel, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the model."""
        raise NotImplementedError("Subclasses must implement forward()")
    
    def get_num_parameters(self) -> int:
        """ Get the total number of trainable parameters. """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path: str):
        """ Save model state dict to file. """
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load(self, path: str, device: str = 'cpu'):
        """ Load model state dict from file. """
        self.load_state_dict(torch.load(path, map_location=device))
        print(f"Model loaded from {path}")
    
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
    
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout: float = 0.5):
        """ Initialize the MLP. """
        super(SimpleMLP, self).__init__()
        
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
        """ Forward pass through the MLP. """
        return self.network(x)


class SimpleCNN(BaseModel):
    """
    Simple Convolutional Neural Network example.
    This is a template/example model architecture.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10
    ):
        """ Initialize the CNN. """
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the CNN. """
        x = self.features(x)
        x = self.classifier(x)
        return x
