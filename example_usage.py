#!/usr/bin/env python3
"""
Example usage of the modular src/ structure.

This script demonstrates how to use the different components of the project
to set up and run a simple training pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Import project modules
from src.config.config import load_config, get_default_config, save_config
from src.models.base_model import SimpleMLP, SimpleCNN
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.utils.logger import setup_logger
from src.utils.metrics import accuracy
from src.utils.visualization import plot_training_history


def main():
    """Main function demonstrating the usage."""
    
    print("=" * 80)
    print("NNDL Project - Example Usage")
    print("=" * 80)
    
    # 1. Configuration Management
    print("\n1. Configuration Management")
    print("-" * 80)
    
    # Get default configuration
    config = get_default_config()
    print(f"Experiment: {config.experiment_name}")
    print(f"Device: {config.device}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    
    # Save configuration for future use
    Path("./configs").mkdir(exist_ok=True)
    save_config(config, "./configs/example_config.yaml")
    
    # 2. Model Creation
    print("\n2. Model Creation")
    print("-" * 80)
    
    # Create a simple MLP model
    mlp_model = SimpleMLP(
        input_dim=784,
        hidden_dims=[256, 128],
        output_dim=10,
        dropout=0.5
    )
    mlp_model.summary()
    
    # Create a simple CNN model
    cnn_model = SimpleCNN(
        in_channels=1,
        num_classes=10
    )
    print(f"CNN Parameters: {cnn_model.get_num_parameters():,}")
    
    # 3. Dummy Dataset Example
    print("\n3. Creating Dummy Dataset")
    print("-" * 80)
    
    # For demonstration, create a simple dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=100):
            self.num_samples = num_samples
            
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Random dummy data
            x = torch.randn(784)  # Flattened 28x28 image
            y = torch.randint(0, 10, (1,)).item()  # Random class
            return x, y
    
    train_dataset = DummyDataset(num_samples=100)
    val_dataset = DummyDataset(num_samples=20)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # 4. Setup Training
    print("\n4. Setting up Training")
    print("-" * 80)
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp_model.parameters(), lr=config.training.learning_rate)
    
    # Create trainer
    trainer = Trainer(
        model=mlp_model,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )
    
    print("Trainer initialized successfully")
    
    # 5. Training (small example)
    print("\n5. Training Model (3 epochs demo)")
    print("-" * 80)
    
    # Create directories for outputs
    Path("./checkpoints").mkdir(exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)
    
    # Define a simple accuracy metric
    def accuracy_metric(predictions, targets):
        pred_classes = predictions.argmax(dim=1)
        correct = (pred_classes == targets).sum().item()
        return correct / targets.size(0)
    
    # Train for a few epochs (small demo)
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=3,
        metric_fn=accuracy_metric,
        save_best=True,
        checkpoint_path="./checkpoints/example_model.pth"
    )
    
    # 6. Evaluation
    print("\n6. Model Evaluation")
    print("-" * 80)
    
    evaluator = Evaluator(mlp_model, device=device)
    results = evaluator.evaluate(val_loader, criterion=criterion)
    
    print(f"Validation Loss: {results['loss']:.4f}")
    
    # Compute accuracy
    acc = accuracy(results['predictions'], results['targets'])
    print(f"Validation Accuracy: {acc:.4f}")
    
    # 7. Visualization
    print("\n7. Visualization")
    print("-" * 80)
    
    # Plot training history
    Path("./results").mkdir(exist_ok=True)
    plot_training_history(
        train_losses=trainer.train_losses,
        val_losses=trainer.val_losses,
        train_metrics=trainer.train_metrics,
        val_metrics=trainer.val_metrics,
        metric_name="Accuracy",
        save_path="./results/training_history.png"
    )
    
    # 8. Model Saving/Loading
    print("\n8. Model Save/Load Demo")
    print("-" * 80)
    
    # Save model
    mlp_model.save("./checkpoints/final_model.pth")
    
    # Load model
    new_model = SimpleMLP(
        input_dim=784,
        hidden_dims=[256, 128],
        output_dim=10
    )
    new_model.load("./checkpoints/final_model.pth", device=device)
    print("Model loaded successfully!")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - configs/example_config.yaml")
    print("  - checkpoints/example_model.pth")
    print("  - checkpoints/final_model.pth")
    print("  - results/training_history.png")


if __name__ == "__main__":
    main()
