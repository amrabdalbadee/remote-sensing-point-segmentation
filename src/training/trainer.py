"""
Training Loop for Point-Supervised Segmentation
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path
from typing import Dict, Optional
import numpy as np

from ..losses import PartialCrossEntropyLoss
from ..point_sampling import batch_sample_points, create_point_sampler
from ..utils.metrics import SegmentationMetrics, RunningMetrics


class Trainer:
    """
    Handles training loop for point-supervised semantic segmentation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        point_sampler=None,
        num_classes: int = 2,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        log_interval: int = 10,
        use_point_supervision: bool = True,
    ):
        """
        Args:
            model: Segmentation model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function (e.g., PartialCrossEntropyLoss)
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            point_sampler: Point sampling strategy
            num_classes: Number of segmentation classes
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_interval: Logging frequency (batches)
            use_point_supervision: If True, use point labels; if False, full supervision
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.point_sampler = point_sampler
        self.num_classes = num_classes
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.use_point_supervision = use_point_supervision
        
        # Metrics tracking
        self.train_metrics = RunningMetrics()
        self.val_metrics = SegmentationMetrics(num_classes=num_classes)
        
        # Best model tracking
        self.best_val_miou = 0.0
        self.current_epoch = 0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Simulate point labels if using point supervision
            if self.use_point_supervision and self.point_sampler is not None:
                point_labels, label_masks = batch_sample_points(
                    masks, self.point_sampler, device=self.device
                )
            else:
                # Full supervision
                point_labels = masks
                label_masks = torch.ones_like(masks, dtype=torch.float32)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Compute loss
            if hasattr(self.criterion, 'forward') and 'label_mask' in \
               self.criterion.forward.__code__.co_varnames:
                loss = self.criterion(outputs, point_labels, label_masks)
            else:
                loss = self.criterion(outputs, point_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            self.train_metrics.update(loss=loss.item())
            
            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{self.train_metrics.get_average('loss'):.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
        
        return self.train_metrics.get_all_averages()
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        self.val_metrics.reset()
        
        val_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss on full masks (not point labels)
            loss = nn.functional.cross_entropy(
                outputs, masks, ignore_index=255
            )
            val_loss += loss.item()
            num_batches += 1
            
            # Update metrics (evaluate on full masks)
            predictions = outputs.argmax(dim=1)
            self.val_metrics.update(predictions, masks)
            
            # Update progress bar
            current_metrics = self.val_metrics.get_metrics()
            pbar.set_postfix({
                'loss': f"{val_loss / num_batches:.4f}",
                'mIoU': f"{current_metrics['miou']:.4f}"
            })
        
        # Get final metrics
        metrics = self.val_metrics.get_metrics()
        metrics['loss'] = val_loss / num_batches
        
        return metrics
    
    def train(self, num_epochs: int, save_best: bool = True):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_best: Save best model based on validation mIoU
        
        Returns:
            dict: Training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_miou': [],
            'val_dice': [],
            'val_pixel_acc': []
        }
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Point supervision: {self.use_point_supervision}")
        if self.use_point_supervision and self.point_sampler:
            print(f"Point sampler: {type(self.point_sampler).__name__}")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['miou'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            print(f"\nEpoch {self.current_epoch}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  Val mIoU:   {val_metrics['miou']:.4f}")
            print(f"  Val Dice:   {val_metrics['mean_dice']:.4f}")
            print(f"  Val Acc:    {val_metrics['pixel_accuracy']:.4f}")
            
            # Save history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_miou'].append(val_metrics['miou'])
            history['val_dice'].append(val_metrics['mean_dice'])
            history['val_pixel_acc'].append(val_metrics['pixel_accuracy'])
            
            # Save best model
            if save_best and val_metrics['miou'] > self.best_val_miou:
                self.best_val_miou = val_metrics['miou']
                self.save_checkpoint(
                    filename='best_model.pth',
                    is_best=True,
                    metrics=val_metrics
                )
                print(f"  ✓ New best model saved! (mIoU: {self.best_val_miou:.4f})")
            
            # Save latest checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    filename=f'checkpoint_epoch_{epoch+1}.pth',
                    metrics=val_metrics
                )
            
            print("-" * 60)
        
        print(f"\nTraining complete!")
        print(f"Best validation mIoU: {self.best_val_miou:.4f}")
        
        return history
    
    def save_checkpoint(self, filename: str, is_best: bool = False, 
                       metrics: Optional[Dict] = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_miou': self.best_val_miou,
            'metrics': metrics
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_miou = checkpoint.get('best_val_miou', 0.0)
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}")


def create_trainer(config: Dict, model: nn.Module, train_loader: DataLoader,
                   val_loader: DataLoader) -> Trainer:
    """
    Factory function to create a Trainer from configuration.
    
    Args:
        config: Configuration dictionary
        model: Segmentation model
        train_loader: Training data loader
        val_loader: Validation data loader
    
    Returns:
        Trainer: Configured trainer instance
    """
    # Create loss function
    criterion = PartialCrossEntropyLoss(
        ignore_index=config.get('ignore_index', 255)
    )
    
    # Create optimizer
    if config.get('optimizer', 'adam').lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.get('learning_rate', 1e-2),
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 1e-5)
        )
    
    # Create scheduler
    scheduler = None
    if config.get('use_scheduler', True):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
    
    # Create point sampler
    point_sampler = None
    if config.get('use_point_supervision', True):
        point_sampler = create_point_sampler(
            strategy=config.get('sampling_strategy', 'random'),
            num_points_per_class=config.get('num_points_per_class', 5)
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        point_sampler=point_sampler,
        num_classes=config.get('num_classes', 2),
        device=config.get('device', 'cuda'),
        checkpoint_dir=config.get('checkpoint_dir', './checkpoints'),
        log_interval=config.get('log_interval', 10),
        use_point_supervision=config.get('use_point_supervision', True)
    )
    
    return trainer