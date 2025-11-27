"""
Evaluation Metrics for Semantic Segmentation
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional


class SegmentationMetrics:
    """
    Computes standard segmentation metrics: IoU, Dice, Pixel Accuracy
    """
    
    def __init__(self, num_classes, ignore_index=255):
        """
        Args:
            num_classes (int): Number of segmentation classes
            ignore_index (int): Label to ignore in metric computation
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, predictions, targets):
        """
        Update metrics with a batch of predictions and targets.
        
        Args:
            predictions (Tensor or np.ndarray): [B, H, W] or [B, C, H, W]
            targets (Tensor or np.ndarray): [B, H, W]
        """
        # Convert to numpy if needed
        if torch.is_tensor(predictions):
            if predictions.dim() == 4:  # [B, C, H, W]
                predictions = predictions.argmax(dim=1)
            predictions = predictions.cpu().numpy()
        
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        
        # Flatten arrays
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Create mask for valid pixels
        valid_mask = (targets != self.ignore_index)
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        # Update confusion matrix
        for pred_class in range(self.num_classes):
            for true_class in range(self.num_classes):
                self.confusion_matrix[true_class, pred_class] += np.sum(
                    (targets == true_class) & (predictions == pred_class)
                )
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics from confusion matrix.
        
        Returns:
            dict: Dictionary containing IoU, mIoU, Dice, Pixel Accuracy, etc.
        """
        metrics = {}
        
        # Per-class IoU
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=0) + 
                self.confusion_matrix.sum(axis=1) - intersection)
        
        iou_per_class = intersection / (union + 1e-10)
        metrics['iou_per_class'] = iou_per_class
        
        # Mean IoU (ignoring classes with no ground truth)
        valid_classes = union > 0
        metrics['miou'] = iou_per_class[valid_classes].mean()
        
        # Per-class Dice coefficient
        dice_per_class = (2 * intersection) / (
            self.confusion_matrix.sum(axis=0) + 
            self.confusion_matrix.sum(axis=1) + 1e-10
        )
        metrics['dice_per_class'] = dice_per_class
        metrics['mean_dice'] = dice_per_class[valid_classes].mean()
        
        # Pixel Accuracy
        total_correct = intersection.sum()
        total_pixels = self.confusion_matrix.sum()
        metrics['pixel_accuracy'] = total_correct / (total_pixels + 1e-10)
        
        # Per-class Precision and Recall
        precision = intersection / (self.confusion_matrix.sum(axis=0) + 1e-10)
        recall = intersection / (self.confusion_matrix.sum(axis=1) + 1e-10)
        
        metrics['precision_per_class'] = precision
        metrics['recall_per_class'] = recall
        metrics['mean_precision'] = precision[valid_classes].mean()
        metrics['mean_recall'] = recall[valid_classes].mean()
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        metrics['f1_per_class'] = f1
        metrics['mean_f1'] = f1[valid_classes].mean()
        
        return metrics
    
    def get_confusion_matrix(self):
        """Return the confusion matrix"""
        return self.confusion_matrix
    
    def print_metrics(self, class_names=None):
        """
        Print formatted metrics.
        
        Args:
            class_names (list): List of class names for display
        """
        metrics = self.get_metrics()
        
        print("\n" + "="*60)
        print("SEGMENTATION METRICS")
        print("="*60)
        
        print(f"\nOverall Metrics:")
        print(f"  Mean IoU:        {metrics['miou']:.4f}")
        print(f"  Mean Dice:       {metrics['mean_dice']:.4f}")
        print(f"  Pixel Accuracy:  {metrics['pixel_accuracy']:.4f}")
        print(f"  Mean Precision:  {metrics['mean_precision']:.4f}")
        print(f"  Mean Recall:     {metrics['mean_recall']:.4f}")
        print(f"  Mean F1:         {metrics['mean_f1']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<20} {'IoU':>8} {'Dice':>8} {'Prec':>8} {'Rec':>8}")
        print("-" * 60)
        
        for i in range(self.num_classes):
            class_name = class_names[i] if class_names else f"Class {i}"
            print(f"{class_name:<20} "
                  f"{metrics['iou_per_class'][i]:>8.4f} "
                  f"{metrics['dice_per_class'][i]:>8.4f} "
                  f"{metrics['precision_per_class'][i]:>8.4f} "
                  f"{metrics['recall_per_class'][i]:>8.4f}")
        
        print("="*60 + "\n")


def compute_iou(predictions, targets, num_classes, ignore_index=255):
    """
    Compute IoU for a single batch (utility function).
    
    Args:
        predictions (Tensor): [B, C, H, W] or [B, H, W]
        targets (Tensor): [B, H, W]
        num_classes (int): Number of classes
        ignore_index (int): Label to ignore
    
    Returns:
        float: Mean IoU
    """
    if predictions.dim() == 4:
        predictions = predictions.argmax(dim=1)
    
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    iou_per_class = []
    
    for cls in range(num_classes):
        pred_mask = (predictions == cls)
        target_mask = (targets == cls)
        ignore_mask = (targets == ignore_index)
        
        # Remove ignored pixels
        pred_mask = pred_mask & ~ignore_mask
        target_mask = target_mask & ~ignore_mask
        
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        
        if union == 0:
            continue
        
        iou = intersection / union
        iou_per_class.append(iou)
    
    return np.mean(iou_per_class) if iou_per_class else 0.0


def compute_dice(predictions, targets, num_classes, ignore_index=255):
    """
    Compute Dice coefficient for a single batch.
    
    Args:
        predictions (Tensor): [B, C, H, W] or [B, H, W]
        targets (Tensor): [B, H, W]
        num_classes (int): Number of classes
        ignore_index (int): Label to ignore
    
    Returns:
        float: Mean Dice coefficient
    """
    if predictions.dim() == 4:
        predictions = predictions.argmax(dim=1)
    
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    dice_per_class = []
    
    for cls in range(num_classes):
        pred_mask = (predictions == cls)
        target_mask = (targets == cls)
        ignore_mask = (targets == ignore_index)
        
        pred_mask = pred_mask & ~ignore_mask
        target_mask = target_mask & ~ignore_mask
        
        intersection = (pred_mask & target_mask).sum()
        total = pred_mask.sum() + target_mask.sum()
        
        if total == 0:
            continue
        
        dice = (2 * intersection) / total
        dice_per_class.append(dice)
    
    return np.mean(dice_per_class) if dice_per_class else 0.0


class RunningMetrics:
    """
    Track metrics over multiple batches with running averages.
    Useful for monitoring during training.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all running metrics"""
        self.metrics = {}
        self.counts = {}
    
    def update(self, **kwargs):
        """
        Update running metrics.
        
        Args:
            **kwargs: Metric names and values
        """
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] += value
            self.counts[key] += 1
    
    def get_average(self, key):
        """Get running average for a metric"""
        if key not in self.metrics:
            return 0.0
        return self.metrics[key] / max(self.counts[key], 1)
    
    def get_all_averages(self):
        """Get all running averages"""
        return {key: self.get_average(key) for key in self.metrics.keys()}
    
    def __str__(self):
        """String representation of current metrics"""
        avg_metrics = self.get_all_averages()
        return ", ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])


# Example usage
if __name__ == "__main__":
    # Test metrics
    num_classes = 5
    metrics = SegmentationMetrics(num_classes=num_classes)
    
    # Simulate some predictions and targets
    predictions = torch.randint(0, num_classes, (4, 256, 256))
    targets = torch.randint(0, num_classes, (4, 256, 256))
    
    metrics.update(predictions, targets)
    metrics.print_metrics()