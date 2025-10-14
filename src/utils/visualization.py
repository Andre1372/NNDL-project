"""
Visualization utilities.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Optional, Tuple


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Optional[List[float]] = None,
    val_metrics: Optional[List[float]] = None,
    metric_name: str = 'Metric',
    save_path: Optional[str] = None
):
    """
    Plot training history (losses and metrics).
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_metrics: Optional training metrics per epoch
        val_metrics: Optional validation metrics per epoch
        metric_name: Name of the metric
        save_path: Optional path to save the figure
    """
    epochs = range(1, len(train_losses) + 1)
    
    if train_metrics is not None and val_metrics is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot metrics
        ax2.plot(epochs, train_metrics, 'b-', label=f'Training {metric_name}')
        ax2.plot(epochs, val_metrics, 'r-', label=f'Validation {metric_name}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(metric_name)
        ax2.set_title(f'Training and Validation {metric_name}')
        ax2.legend()
        ax2.grid(True)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Optional list of class names
        normalize: Whether to normalize the confusion matrix
        save_path: Optional path to save the figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot confusion matrix using matplotlib's imshow
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks and labels
    if class_names:
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
    else:
        ax.set_xticks(np.arange(cm.shape[1]))
        ax.set_yticks(np.arange(cm.shape[0]))
    
    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def visualize_predictions(
    images: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: Optional[List[str]] = None,
    num_images: int = 16,
    save_path: Optional[str] = None
):
    """
    Visualize model predictions on images.
    
    Args:
        images: Batch of images
        predictions: Model predictions
        targets: Ground truth labels
        class_names: Optional list of class names
        num_images: Number of images to display
        save_path: Optional path to save the figure
    """
    num_images = min(num_images, images.size(0))
    pred_classes = predictions.argmax(dim=1)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_images):
        img = images[i].cpu().numpy()
        
        # Handle different image formats
        if img.shape[0] == 1:  # Grayscale
            img = img.squeeze()
            axes[i].imshow(img, cmap='gray')
        elif img.shape[0] == 3:  # RGB
            img = np.transpose(img, (1, 2, 0))
            axes[i].imshow(img)
        else:
            axes[i].imshow(img.squeeze())
        
        pred_label = class_names[pred_classes[i]] if class_names else pred_classes[i]
        true_label = class_names[targets[i]] if class_names else targets[i]
        
        color = 'green' if pred_classes[i] == targets[i] else 'red'
        axes[i].set_title(f'Pred: {pred_label}\nTrue: {true_label}', color=color)
        axes[i].axis('off')
    
    # Hide remaining subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_learning_rate(
    learning_rates: List[float],
    save_path: Optional[str] = None
):
    """
    Plot learning rate schedule.
    
    Args:
        learning_rates: List of learning rates per epoch
        save_path: Optional path to save the figure
    """
    epochs = range(1, len(learning_rates) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, learning_rates, 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
