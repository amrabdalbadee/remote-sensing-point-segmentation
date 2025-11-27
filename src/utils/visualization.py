"""
Visualization utilities for segmentation results
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
import seaborn as sns
from typing import Optional, List


def plot_training_history(history, save_path=None, figsize=(15, 5)):
    """
    Plot training history curves.
    
    Args:
        history (dict): Dictionary containing training metrics
        save_path (str): Path to save the plot
        figsize (tuple): Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot mIoU
    axes[1].plot(history['val_miou'], label='Val mIoU', color='green', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('mIoU', fontsize=12)
    axes[1].set_title('Validation mIoU', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot other metrics
    axes[2].plot(history['val_dice'], label='Dice', linewidth=2)
    axes[2].plot(history['val_pixel_acc'], label='Pixel Acc', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Score', fontsize=12)
    axes[2].set_title('Validation Metrics', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def visualize_segmentation(image, ground_truth, prediction, point_labels=None,
                          class_names=None, figsize=(20, 5), save_path=None):
    """
    Visualize image, ground truth, prediction, and optionally point labels.
    
    Args:
        image (np.ndarray or Tensor): [H, W, 3] or [3, H, W] RGB image
        ground_truth (np.ndarray or Tensor): [H, W] ground truth mask
        prediction (np.ndarray or Tensor): [H, W] or [C, H, W] prediction
        point_labels (np.ndarray or Tensor): [H, W] sparse point labels (optional)
        class_names (list): List of class names
        figsize (tuple): Figure size
        save_path (str): Path to save the plot
    """
    # Convert tensors to numpy
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.cpu().numpy()
    if torch.is_tensor(prediction):
        if prediction.dim() == 3:  # [C, H, W]
            prediction = prediction.argmax(dim=0)
        prediction = prediction.cpu().numpy()
    if point_labels is not None and torch.is_tensor(point_labels):
        point_labels = point_labels.cpu().numpy()
    
    # Ensure image is [H, W, 3]
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    # Normalize image to [0, 1] if needed
    if image.max() > 1:
        image = image / 255.0
    
    # Create subplots
    num_plots = 4 if point_labels is not None else 3
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    
    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Plot ground truth
    axes[1].imshow(ground_truth, cmap='tab20')
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Plot prediction
    axes[2].imshow(prediction, cmap='tab20')
    axes[2].set_title('Prediction', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Plot point labels if provided
    if point_labels is not None:
        # Show image with point overlays
        axes[3].imshow(image)
        # Overlay points
        point_mask = point_labels != 255  # Assuming 255 is ignore index
        if point_mask.any():
            y_coords, x_coords = np.where(point_mask)
            axes[3].scatter(x_coords, y_coords, c=point_labels[point_mask], 
                          cmap='tab20', s=50, edgecolors='white', linewidth=1)
        axes[3].set_title('Point Annotations', fontsize=14, fontweight='bold')
        axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def visualize_batch_predictions(images, ground_truths, predictions, 
                                num_samples=4, class_names=None, 
                                save_path=None):
    """
    Visualize multiple samples from a batch.
    
    Args:
        images (Tensor): [B, 3, H, W]
        ground_truths (Tensor): [B, H, W]
        predictions (Tensor): [B, C, H, W] or [B, H, W]
        num_samples (int): Number of samples to visualize
        class_names (list): List of class names
        save_path (str): Path to save the plot
    """
    num_samples = min(num_samples, images.shape[0])
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Get image
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        if img.max() > 1:
            img = img / 255.0
        
        # Get ground truth and prediction
        gt = ground_truths[i].cpu().numpy()
        if predictions.dim() == 4:
            pred = predictions[i].argmax(dim=0).cpu().numpy()
        else:
            pred = predictions[i].cpu().numpy()
        
        # Plot
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Sample {i+1}: Image', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gt, cmap='tab20')
        axes[i, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred, cmap='tab20')
        axes[i, 2].set_title('Prediction', fontsize=12, fontweight='bold')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Batch visualization saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(confusion_matrix, class_names=None, 
                         normalize=True, save_path=None, figsize=(10, 8)):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        confusion_matrix (np.ndarray): [num_classes, num_classes] confusion matrix
        class_names (list): List of class names
        normalize (bool): Normalize by row (true label)
        save_path (str): Path to save the plot
        figsize (tuple): Figure size
    """
    if normalize:
        cm = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-10)
    else:
        cm = confusion_matrix
    
    plt.figure(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', square=True, linewidths=0.5,
                xticklabels=class_names if class_names else range(len(cm)),
                yticklabels=class_names if class_names else range(len(cm)),
                cbar_kws={'label': 'Normalized Count' if normalize else 'Count'})
    
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_class_metrics(metrics_dict, class_names=None, save_path=None, figsize=(12, 6)):
    """
    Plot per-class metrics as bar charts.
    
    Args:
        metrics_dict (dict): Dictionary with 'iou_per_class', 'dice_per_class', etc.
        class_names (list): List of class names
        save_path (str): Path to save the plot
        figsize (tuple): Figure size
    """
    num_classes = len(metrics_dict['iou_per_class'])
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    x = np.arange(num_classes)
    width = 0.35
    
    # Plot IoU
    axes[0].bar(x, metrics_dict['iou_per_class'], width, label='IoU', alpha=0.8)
    axes[0].axhline(y=metrics_dict['miou'], color='r', linestyle='--', 
                    label=f"Mean IoU: {metrics_dict['miou']:.3f}")
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('IoU', fontsize=12)
    axes[0].set_title('Per-Class IoU', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot Precision and Recall
    axes[1].bar(x - width/2, metrics_dict['precision_per_class'], width, 
                label='Precision', alpha=0.8)
    axes[1].bar(x + width/2, metrics_dict['recall_per_class'], width,
                label='Recall', alpha=0.8)
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Per-Class Precision & Recall', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class metrics plot saved to {save_path}")
    
    plt.show()


def visualize_point_sampling_comparison(mask, samplers_dict, figsize=(20, 5), 
                                       save_path=None):
    """
    Compare different point sampling strategies side by side.
    
    Args:
        mask (np.ndarray): [H, W] segmentation mask
        samplers_dict (dict): Dictionary of {name: sampler} pairs
        figsize (tuple): Figure size
        save_path (str): Path to save the plot
    """
    num_strategies = len(samplers_dict) + 1  # +1 for original mask
    fig, axes = plt.subplots(1, num_strategies, figsize=figsize)
    
    # Plot original mask
    axes[0].imshow(mask, cmap='tab20')
    axes[0].set_title('Original Mask', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Plot each sampling strategy
    for idx, (name, sampler) in enumerate(samplers_dict.items(), start=1):
        point_labels, label_mask = sampler.sample(mask)
        
        # Visualize points on mask
        axes[idx].imshow(mask, cmap='tab20', alpha=0.5)
        
        # Overlay points
        y_coords, x_coords = np.where(label_mask > 0)
        axes[idx].scatter(x_coords, y_coords, c='red', s=30, 
                         edgecolors='white', linewidth=1, marker='x')
        
        num_points = (label_mask > 0).sum()
        axes[idx].set_title(f'{name}\n({num_points} points)', 
                          fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Point sampling comparison saved to {save_path}")
    
    plt.show()


def create_color_legend(class_names, cmap='tab20', save_path=None, figsize=(8, 6)):
    """
    Create a color legend for segmentation classes.
    
    Args:
        class_names (list): List of class names
        cmap (str): Colormap name
        save_path (str): Path to save the plot
        figsize (tuple): Figure size
    """
    num_classes = len(class_names)
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, num_classes))
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    for i, (name, color) in enumerate(zip(class_names, colors)):
        ax.add_patch(plt.Rectangle((0, i), 1, 0.8, facecolor=color))
        ax.text(1.2, i + 0.4, name, va='center', fontsize=12)
    
    ax.set_xlim(0, 3)
    ax.set_ylim(0, num_classes)
    ax.invert_yaxis()
    ax.set_title('Class Color Legend', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Color legend saved to {save_path}")
    
    plt.show()


# Example usage
if __name__ == "__main__":
    # Test visualization functions
    import torch
    
    # Create dummy data
    image = torch.randn(3, 256, 256)
    ground_truth = torch.randint(0, 5, (256, 256))
    prediction = torch.randn(5, 256, 256)
    
    # Visualize
    visualize_segmentation(image, ground_truth, prediction,
                          class_names=['Background', 'Building', 'Road', 'Water', 'Vegetation'])