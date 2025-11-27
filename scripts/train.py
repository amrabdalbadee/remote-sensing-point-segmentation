"""
Main Training Script for Point-Supervised Segmentation
Usage: python scripts/train.py --config config/default.yaml
"""
import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.unet import UNet, get_unet
from src.losses import PartialCrossEntropyLoss
from src.point_sampling import create_point_sampler
from src.training.trainer import create_trainer
from src.utils.visualization import plot_training_history
from torch.utils.data import DataLoader


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config, device=None):
    """
    Create train and validation dataloaders.
    
    NOTE: You need to implement your dataset class based on your chosen dataset.
    This is a placeholder that should be replaced with actual dataset loading.
    """
    # Example for LoveDA dataset
    from src.datasets.dataset import LoveDADataset
    
    # Training dataset
    train_dataset = LoveDADataset(
        root_dir=config['data']['root_dir'],
        split='train',
        img_size=config['data']['img_size'],
        augmentation=config['data'].get('augmentation', True)
    )
    
    # Validation dataset
    val_dataset = LoveDADataset(
        root_dir=config['data']['root_dir'],
        split='val',
        img_size=config['data']['img_size'],
        augmentation=False
    )
    
    # Create dataloaders
    # MPS doesn't support pin_memory, so disable it for MPS and CPU
    if device is not None:
        pin_memory = device.type != 'mps' and device.type != 'cpu'
    else:
        pin_memory = False
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def create_model(config):
    """Create segmentation model"""
    model_type = config['model']['type'].lower()
    num_classes = config['model']['num_classes']
    
    if model_type == 'unet':
        model = UNet(
            in_channels=3,
            num_classes=num_classes,
            bilinear=config['model'].get('bilinear', True),
            base_channels=config['model'].get('base_channels', 64)
        )
    elif model_type == 'light_unet':
        model = get_unet('light', in_channels=3, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def main(args):
    """Main training function"""
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config.get('seed', 42))
    
    # Set device (prioritize MPS for M1 Macs, then CUDA, then CPU)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    config['training']['device'] = str(device)
    print(f"Using device: {device}")
    
    # Create output directories
    exp_dir = Path(config['experiment']['output_dir'])
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'visualizations').mkdir(exist_ok=True)
    
    # Save config
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print("\n" + "="*60)
    print("EXPERIMENT CONFIGURATION")
    print("="*60)
    print(f"Experiment name: {config['experiment']['name']}")
    print(f"Model: {config['model']['type']}")
    print(f"Number of classes: {config['model']['num_classes']}")
    print(f"Point supervision: {config['training'].get('use_point_supervision', True)}")
    if config['training'].get('use_point_supervision', True):
        print(f"Sampling strategy: {config['training']['sampling_strategy']}")
        print(f"Points per class: {config['training']['num_points_per_class']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Number of epochs: {config['training']['num_epochs']}")
    print("="*60 + "\n")
    
    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader = create_dataloaders(config, device)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create trainer configuration
    trainer_config = {
        'num_classes': config['model']['num_classes'],
        'device': device,
        'checkpoint_dir': exp_dir / 'checkpoints',
        'log_interval': config['training'].get('log_interval', 10),
        'use_point_supervision': config['training'].get('use_point_supervision', True),
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training'].get('weight_decay', 1e-5),
        'optimizer': config['training'].get('optimizer', 'adam'),
        'use_scheduler': config['training'].get('use_scheduler', True),
        'sampling_strategy': config['training'].get('sampling_strategy', 'random'),
        'num_points_per_class': config['training'].get('num_points_per_class', 5),
        'ignore_index': config['data'].get('ignore_index', 255)
    }
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = create_trainer(trainer_config, model, train_loader, val_loader)
    
    # Load checkpoint if resuming
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    print("\nStarting training...\n")
    history = trainer.train(
        num_epochs=config['training']['num_epochs'],
        save_best=True
    )
    
    # Save training history
    print("\nSaving training history...")
    history_path = exp_dir / 'training_history.yaml'
    with open(history_path, 'w') as f:
        yaml.dump(history, f)
    
    # Plot training curves
    print("Generating training plots...")
    plot_training_history(
        history,
        save_path=exp_dir / 'visualizations' / 'training_curves.png'
    )
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Best validation mIoU: {trainer.best_val_miou:.4f}")
    print(f"Outputs saved to: {exp_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train point-supervised segmentation model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    main(args)