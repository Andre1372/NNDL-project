"""
Example usage of the modular src/ structure.

This script demonstrates how to use the different components of the project
to set up and run a simple training pipeline using PyTorch Lightning.
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Import project modules
from data.dataset import PolyDataset
from data.transforms import ToTensor
from models.base_model import SimpleMLP
from training.trainer import LightningTrainer

def poly_model(x, beta, noise_std=0):
    """
    INPUT
        x: x vector
        beta: polynomial parameters
        noise_std: enable noisy sampling (gaussian noise, zero mean, noise_std std)
    """
    pol_order = len(beta)
    x_matrix = np.array([x**i for i in range(pol_order)]).transpose()
    y = np.matmul(x_matrix, beta)
    noise = np.random.randn(len(y)) * noise_std
    return y + noise


def main():
    """Main function demonstrating the usage."""
    
    print("=" * 80)
    print("NNDL Project - Example Usage")
    print("=" * 80)

    # 1. Data Preparation
    print("\n1. Data Preparation")
    print("-" * 80)

    beta_true = [8.6, -16.2, 12.2, -2.8, 0.2]  # True polynomial coefficients
    noise_std = 0.2
    np.random.seed(4)

    ### Train data
    num_train_points = 200
    x_train = np.random.rand(num_train_points)*5
    y_train = poly_model(x_train, beta_true, noise_std)
    train_dataset = PolyDataset(inputs=x_train.reshape(-1,1), targets=y_train.reshape(-1,1), transform=ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True)

    ### Validation data
    num_val_points = 20
    x_val = np.random.rand(num_val_points)*5
    y_val = poly_model(x_val, beta_true, noise_std)
    val_dataset = PolyDataset(inputs=x_val.reshape(-1,1), targets=y_val.reshape(-1,1), transform=ToTensor())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    # 2. Model Creation
    print("\n2. Model Creation")
    print("-" * 80)
    
    # Create a simple MLP model with Lightning
    mlp_model = SimpleMLP(
        input_dim=1,
        hidden_dims=[50, 100, 50],
        output_dim=1,
        dropout=0,
        learning_rate=1e-3,
        criterion=nn.MSELoss()
    )
    mlp_model.summary()

    # 3. Training
    print("\n3. Setting up Training with PyTorch Lightning")
    print("-" * 80)
    
    # Create Lightning trainer
    lightning_trainer = LightningTrainer(
        max_epochs=50,
        enable_checkpointing=True,
        enable_early_stopping=False,
        log_every_n_steps=4,

        enable_logging=True,
        experiment_name='demo',
        overwrite_last=False,
    )
    
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

    print("\nTo view training logs, run:")
    print("  tensorboard --logdir=lightning_logs/")
    
    plt.figure(figsize=(12,8))
    x_highres = np.linspace(0,1,1000)*5
    plt.plot(x_highres, poly_model(x_highres, beta_true), color='b', ls='--', label='True data model')
    plt.plot(x_highres, mlp_model(torch.tensor(x_highres.reshape(-1,1), dtype=torch.float32)).detach().numpy(), color='r', ls='-', label='MLP model prediction')
    plt.plot(x_train, y_train, color='g', ls='', marker='.', label='Train data points')
    plt.plot(x_val, y_val, color='m', ls='', marker='.', label='Validation data points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    



if __name__ == "__main__":
    main()