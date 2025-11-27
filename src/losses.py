import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialCrossEntropyLoss(nn.Module):
    """
    Partial Cross-Entropy Loss for Point-Supervised Segmentation
    Computes cross-entropy loss only on labeled pixels (point annotations).
    Unlabeled pixels are ignored during loss computation.
    """
    
    def __init__(self, ignore_index=255, reduction='mean', weight=None):
        """
        Args:
            ignore_index (int): Label value to ignore in loss computation
            reduction (str): 'mean', 'sum', or 'none'
            weight (Tensor): Manual rescaling weight for each class
        """
        super(PartialCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.weight = weight
        
    def forward(self, predictions, targets, label_mask=None):
        """
        Args:
            predictions (Tensor): [B, C, H, W] - Model predictions (logits)
            targets (Tensor): [B, H, W] - Sparse ground truth labels
            label_mask (Tensor): [B, H, W] - Binary mask (1=labeled, 0=unlabeled)
                                 If None, uses targets != ignore_index
        
        Returns:
            loss (Tensor): Computed partial cross-entropy loss
        """
        batch_size, num_classes, height, width = predictions.shape
        
        # Create label mask if not provided
        if label_mask is None:
            label_mask = (targets != self.ignore_index).float()
        
        # Flatten spatial dimensions
        predictions_flat = predictions.permute(0, 2, 3, 1).reshape(-1, num_classes)
        targets_flat = targets.reshape(-1)
        label_mask_flat = label_mask.reshape(-1)
        
        # Compute cross-entropy for all pixels
        loss_per_pixel = F.cross_entropy(
            predictions_flat,
            targets_flat,
            weight=self.weight,
            reduction='none',
            ignore_index=self.ignore_index
        )
        
        # Apply mask to keep only labeled pixels
        masked_loss = loss_per_pixel * label_mask_flat
        
        # Compute final loss based on reduction type
        if self.reduction == 'mean':
            # Average over labeled pixels only
            num_labeled = label_mask_flat.sum()
            if num_labeled > 0:
                loss = masked_loss.sum() / num_labeled
            else:
                loss = masked_loss.sum()  # Returns 0 if no labeled pixels
        elif self.reduction == 'sum':
            loss = masked_loss.sum()
        else:  # 'none'
            loss = masked_loss.reshape(batch_size, height, width)
        
        return loss


class WeightedPartialCrossEntropyLoss(nn.Module):
    """
    Partial CE Loss with automatic class weighting based on labeled pixels.
    Useful for handling class imbalance in point annotations.
    """
    
    def __init__(self, num_classes, ignore_index=255, reduction='mean'):
        super(WeightedPartialCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, predictions, targets, label_mask=None):
        """
        Args:
            predictions (Tensor): [B, C, H, W]
            targets (Tensor): [B, H, W]
            label_mask (Tensor): [B, H, W]
        """
        # Compute class frequencies from labeled pixels
        if label_mask is None:
            label_mask = (targets != self.ignore_index).float()
        
        class_weights = self._compute_class_weights(targets, label_mask)
        
        # Use standard partial CE loss with computed weights
        criterion = PartialCrossEntropyLoss(
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            weight=class_weights
        )
        
        return criterion(predictions, targets, label_mask)
    
    def _compute_class_weights(self, targets, label_mask):
        """Compute inverse frequency weights for labeled pixels"""
        device = targets.device
        class_counts = torch.zeros(self.num_classes, device=device)
        
        # Count labeled pixels per class
        for c in range(self.num_classes):
            class_mask = (targets == c) & (label_mask.bool())
            class_counts[c] = class_mask.sum()
        
        # Compute inverse frequency weights
        # Add small epsilon to avoid division by zero
        total_labeled = class_counts.sum()
        if total_labeled > 0:
            class_weights = total_labeled / (class_counts + 1e-6)
            # Normalize weights
            class_weights = class_weights / class_weights.sum() * self.num_classes
        else:
            class_weights = torch.ones(self.num_classes, device=device)
        
        return class_weights


class CombinedLoss(nn.Module):
    """
    Combines partial CE loss with auxiliary losses (e.g., Dice, consistency).
    Useful for improving segmentation quality with sparse supervision.
    """
    
    def __init__(self, ce_weight=1.0, dice_weight=0.0, ignore_index=255):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.partial_ce = PartialCrossEntropyLoss(ignore_index=ignore_index)
        
    def forward(self, predictions, targets, label_mask=None):
        """Compute combined loss"""
        loss = 0.0
        
        # Partial cross-entropy loss
        if self.ce_weight > 0:
            ce_loss = self.partial_ce(predictions, targets, label_mask)
            loss += self.ce_weight * ce_loss
        
        # Can add Dice loss or other auxiliary losses here
        if self.dice_weight > 0:
            dice_loss = self._dice_loss(predictions, targets, label_mask)
            loss += self.dice_weight * dice_loss
        
        return loss
    
    def _dice_loss(self, predictions, targets, label_mask):
        """Compute Dice loss on labeled pixels"""
        # Placeholder for Dice loss implementation
        # Can be extended based on requirements
        return torch.tensor(0.0, device=predictions.device)
