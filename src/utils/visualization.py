
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Optional


def plot_training_history(
    train_losses: List[float],          # training losses per epoch
    val_losses: List[float],            # validation losses per epoch
    train_metrics: Optional[List[float]] = None,    # training metrics per epoch
    val_metrics: Optional[List[float]] = None,      # validation metrics per epoch
    metric_name: str = 'Metric',        # name of the metric
    save_path: Optional[str] = None     # path to save the figure
):
    """ Plot training history (losses and metrics). """

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
    cm: np.ndarray,                             # confusion matrix
    class_names: Optional[List[str]] = None,    # optional class names
    normalize: bool = False,                    # whether to normalize the matrix
    save_path: Optional[str] = None             # path to save the figure
):
    """ Plot confusion matrix. """

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
