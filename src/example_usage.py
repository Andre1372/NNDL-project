"""
Example usage of the modular src/ structure.

This script demonstrates how to use the different components of the project
to set up and run a simple training pipeline using PyTorch Lightning.
"""

import torch
import torch.nn as nn
from pathlib import Path

# Import project modules
from config.config import load_config
from models.base_model import SimpleMLP
from training.trainer import LightningTrainer


def main():
    """Main function demonstrating the usage."""
    
    print("=" * 80)
    print("NNDL Project - Example Usage")
    print("=" * 80)
    
    # 1. Configuration Management
    print("\n1. Configuration Management")
    print("-" * 80)
    
    # Get default configuration
    config = load_config("configs/default_config.json")
    print(f"Experiment: {config.experiment_name}")
    print(f"Device: {config.device}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")

    # 2. Model Creation
    print("\n2. Model Creation")
    print("-" * 80)
    
    # Create a simple MLP model with Lightning
    mlp_model = SimpleMLP(
        input_dim=784,
        hidden_dims=[256, 128],
        output_dim=10,
        dropout=0.5,
        learning_rate=config.training.learning_rate,
        criterion=nn.CrossEntropyLoss()
    )
    mlp_model.summary()
    
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
    print("\n4. Setting up Training with PyTorch Lightning")
    print("-" * 80)
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        accelerator = "gpu"
    elif torch.backends.mps.is_available():
        device = "mps"
        accelerator = "mps"
    else:
        device = "cpu"
        accelerator = "cpu"
    
    print(f"Using device: {device}")
    
    # Create Lightning trainer
    lightning_trainer = LightningTrainer(
        max_epochs=3,
        accelerator=accelerator,
        devices=1,
        enable_checkpointing=True,
        checkpoint_dir="checkpoints",
        enable_early_stopping=False
    )
    
    print("Lightning Trainer initialized successfully")
    
    # 5. Training (small example)
    print("\n5. Training Model with Lightning (3 epochs demo)")
    print("-" * 80)
    
    # Create directories for outputs
    Path("checkpoints").mkdir(exist_ok=True)
    
    # Train for a few epochs (small demo)
    lightning_trainer.fit(
        model=mlp_model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # 6. Evaluation
    print("\n6. Model Evaluation")
    print("-" * 80)
    
    # Test the model with Lightning
    lightning_trainer.test(mlp_model, test_loader=val_loader)
    
    # 7. Visualization
    print("\n7. Visualization")
    print("-" * 80)
    
    # Note: With Lightning, metrics are logged automatically
    # You can access them from the trainer's logged metrics
    # For visualization, you can use TensorBoard or the logged metrics
    print("Training metrics are logged by PyTorch Lightning")
    print("Use TensorBoard to visualize: tensorboard --logdir=lightning_logs/")
    
    # For custom visualization, you'd need to extract metrics from logs
    # or use callbacks during training
    
    # 8. Model Saving/Loading
    print("\n8. Model Save/Load Demo")
    print("-" * 80)
    
    # Save model
    mlp_model.save("checkpoints/final_model.pth")
    
    # Load model
    new_model = SimpleMLP(
        input_dim=784,
        hidden_dims=[256, 128],
        output_dim=10,
        learning_rate=config.training.learning_rate
    )
    new_model.load("checkpoints/final_model.pth", device=device)
    print("Model loaded successfully!")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - checkpoints/best-*.ckpt (Lightning checkpoint)")
    print("  - checkpoints/last.ckpt (Last checkpoint)")
    print("  - checkpoints/final_model.pth (State dict)")
    print("  - lightning_logs/ (TensorBoard logs)")
    print("\nTo view training logs, run:")
    print("  tensorboard --logdir=lightning_logs/")


if __name__ == "__main__":
    main()
