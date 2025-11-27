"""
Model Evaluation Script for Point-Supervised Segmentation

Evaluates a trained model on test/validation set and generates comprehensive results.

Usage:
    python scripts/evaluate.py \
        --checkpoint experiments/baseline/checkpoints/best_model.pth \
        --config config/default.yaml \
        --split test \
        --output_dir evaluation_results
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.unet import UNet, get_unet
from src.utils.metrics import SegmentationMetrics, compute_iou, compute_dice
from src.utils.visualization import (
    plot_confusion_matrix, plot_class_metrics,
    visualize_segmentation, visualize_batch_predictions
)

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11


class ModelEvaluator:
    """Comprehensive model evaluation with metrics and visualizations"""
    
    def __init__(self, model, test_loader, device='cuda', 
                 num_classes=7, class_names=None, output_dir='evaluation_results'):
        """
        Args:
            model: Trained segmentation model
            test_loader: DataLoader for test set
            device: Device to run evaluation on
            num_classes: Number of segmentation classes
            class_names: List of class names for visualization
            output_dir: Directory to save results
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        
        # Create output directory structure
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        (self.output_dir / 'predictions').mkdir(exist_ok=True)
        (self.output_dir / 'metrics').mkdir(exist_ok=True)
        
        # Initialize metrics
        self.metrics = SegmentationMetrics(num_classes=num_classes)
        
        # Results storage
        self.all_predictions = []
        self.all_targets = []
        
        print(f"\n{'='*70}")
        print("MODEL EVALUATOR INITIALIZED")
        print(f"{'='*70}")
        print(f"Device: {device}")
        print(f"Number of classes: {num_classes}")
        print(f"Test samples: {len(test_loader.dataset)}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*70}\n")
    
    @torch.no_grad()
    def evaluate(self, save_predictions=False, num_visualizations=10):
        """
        Run complete evaluation
        
        Args:
            save_predictions: Whether to save prediction visualizations
            num_visualizations: Number of samples to visualize
        
        Returns:
            dict: Complete evaluation results
        """
        print("Starting evaluation...\n")
        
        self.model.eval()
        self.metrics.reset()
        
        # Track additional statistics
        inference_times = []
        sample_indices_to_visualize = np.linspace(
            0, len(self.test_loader.dataset) - 1, 
            num_visualizations, dtype=int
        )
        current_idx = 0
        
        # Progress bar
        pbar = tqdm(self.test_loader, desc="Evaluating")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Measure inference time
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = datetime.now()
            predictions = self.model(images)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            inference_time = (datetime.now() - start_time).total_seconds()
            inference_times.append(inference_time / images.shape[0])  # Per image
            
            # Get predicted classes
            pred_classes = predictions.argmax(dim=1)
            
            # Update metrics
            self.metrics.update(pred_classes, masks)
            
            # Store for later analysis
            self.all_predictions.append(pred_classes.cpu())
            self.all_targets.append(masks.cpu())
            
            # Visualize selected samples
            if save_predictions:
                for i in range(images.shape[0]):
                    if current_idx in sample_indices_to_visualize:
                        self._save_prediction_visualization(
                            images[i], masks[i], pred_classes[i], current_idx
                        )
                    current_idx += 1
            
            # Update progress bar
            current_metrics = self.metrics.get_metrics()
            pbar.set_postfix({
                'mIoU': f"{current_metrics['miou']:.4f}",
                'Dice': f"{current_metrics['mean_dice']:.4f}"
            })
        
        print("\n✓ Evaluation complete!\n")
        
        # Compile results
        results = self._compile_results(inference_times)
        
        # Generate visualizations
        print("Generating visualizations...")
        self._generate_visualizations()
        
        # Save results
        print("Saving results...")
        self._save_results(results)
        
        print(f"\n{'='*70}")
        print("EVALUATION COMPLETE")
        print(f"{'='*70}")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*70}\n")
        
        return results
    
    def _compile_results(self, inference_times):
        """Compile all evaluation metrics"""
        
        # Get metrics from confusion matrix
        metrics = self.metrics.get_metrics()
        
        # Add inference statistics
        metrics['inference_time_mean'] = float(np.mean(inference_times))
        metrics['inference_time_std'] = float(np.std(inference_times))
        metrics['inference_fps'] = float(1.0 / np.mean(inference_times))
        
        # Add per-class metrics as lists (for JSON serialization)
        metrics['iou_per_class'] = metrics['iou_per_class'].tolist()
        metrics['dice_per_class'] = metrics['dice_per_class'].tolist()
        metrics['precision_per_class'] = metrics['precision_per_class'].tolist()
        metrics['recall_per_class'] = metrics['recall_per_class'].tolist()
        
        # Add class names
        metrics['class_names'] = self.class_names
        
        # Add timestamp
        metrics['evaluation_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return metrics
    
    def _generate_visualizations(self):
        """Generate all visualization figures"""
        
        # 1. Confusion Matrix
        print("  - Generating confusion matrix...")
        cm = self.metrics.get_confusion_matrix()
        plot_confusion_matrix(
            cm, 
            class_names=self.class_names,
            normalize=True,
            save_path=self.output_dir / 'figures' / 'confusion_matrix.png',
            figsize=(10, 8)
        )
        plt.close('all')
        
        # 2. Per-Class Metrics
        print("  - Generating per-class metrics...")
        metrics = self.metrics.get_metrics()
        # Convert back to numpy for visualization
        metrics_viz = {
            'iou_per_class': np.array(metrics['iou_per_class']),
            'dice_per_class': np.array(metrics['dice_per_class']),
            'precision_per_class': np.array(metrics['precision_per_class']),
            'recall_per_class': np.array(metrics['recall_per_class']),
            'miou': metrics['miou']
        }
        plot_class_metrics(
            metrics_viz,
            class_names=self.class_names,
            save_path=self.output_dir / 'figures' / 'per_class_metrics.png',
            figsize=(14, 6)
        )
        plt.close('all')
        
        # 3. Metrics Summary Bar Chart
        print("  - Generating metrics summary...")
        self._plot_metrics_summary(metrics)
        plt.close('all')
        
        print("✓ Visualizations generated!\n")
    
    def _plot_metrics_summary(self, metrics):
        """Create summary bar chart of main metrics"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = ['mIoU', 'Mean Dice', 'Pixel Accuracy', 
                       'Mean Precision', 'Mean Recall']
        metric_values = [
            metrics['miou'],
            metrics['mean_dice'],
            metrics['pixel_accuracy'],
            metrics['mean_precision'],
            metrics['mean_recall']
        ]
        
        colors = ['#2E86AB', '#F18F01', '#C73E1D', '#6A994E', '#A23B72']
        bars = ax.bar(metric_names, metric_values, color=colors, 
                     alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.4f}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax.set_title('Evaluation Metrics Summary', fontsize=15, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'metrics_summary.png', 
                   dpi=300, bbox_inches='tight')
    
    def _save_prediction_visualization(self, image, ground_truth, prediction, idx):
        """Save individual prediction visualization"""
        
        save_path = self.output_dir / 'predictions' / f'sample_{idx:04d}.png'
        
        visualize_segmentation(
            image, 
            ground_truth, 
            prediction,
            class_names=self.class_names,
            save_path=save_path,
            figsize=(15, 5)
        )
        plt.close('all')
    
    def _save_results(self, results):
        """Save results in multiple formats"""
        
        # 1. Save as JSON
        json_path = self.output_dir / 'metrics' / 'evaluation_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  ✓ JSON results: {json_path}")
        
        # 2. Save as YAML
        yaml_path = self.output_dir / 'metrics' / 'evaluation_results.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False, sort_keys=False)
        print(f"  ✓ YAML results: {yaml_path}")
        
        # 3. Save detailed text report
        report_path = self.output_dir / 'metrics' / 'evaluation_report.txt'
        self._save_text_report(results, report_path)
        print(f"  ✓ Text report: {report_path}")
        
        # 4. Save per-class results as CSV
        csv_path = self.output_dir / 'metrics' / 'per_class_results.csv'
        self._save_csv_results(results, csv_path)
        print(f"  ✓ CSV results: {csv_path}")
    
    def _save_text_report(self, results, path):
        """Save detailed text report"""
        
        with open(path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Evaluation Date: {results['evaluation_time']}\n")
            f.write(f"Number of Classes: {self.num_classes}\n")
            f.write(f"Test Samples: {len(self.test_loader.dataset)}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("OVERALL METRICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Mean IoU (mIoU):          {results['miou']:.6f}\n")
            f.write(f"Mean Dice Coefficient:    {results['mean_dice']:.6f}\n")
            f.write(f"Pixel Accuracy:           {results['pixel_accuracy']:.6f}\n")
            f.write(f"Mean Precision:           {results['mean_precision']:.6f}\n")
            f.write(f"Mean Recall:              {results['mean_recall']:.6f}\n")
            f.write(f"Mean F1 Score:            {results['mean_f1']:.6f}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("INFERENCE STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Mean Inference Time:      {results['inference_time_mean']:.4f} seconds/image\n")
            f.write(f"Std Inference Time:       {results['inference_time_std']:.4f} seconds\n")
            f.write(f"Inference FPS:            {results['inference_fps']:.2f} images/second\n\n")
            
            f.write("-"*80 + "\n")
            f.write("PER-CLASS METRICS\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Class':<20} {'IoU':>10} {'Dice':>10} {'Precision':>10} {'Recall':>10}\n")
            f.write("-"*80 + "\n")
            
            for i, class_name in enumerate(self.class_names):
                f.write(f"{class_name:<20} "
                       f"{results['iou_per_class'][i]:>10.6f} "
                       f"{results['dice_per_class'][i]:>10.6f} "
                       f"{results['precision_per_class'][i]:>10.6f} "
                       f"{results['recall_per_class'][i]:>10.6f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
    
    def _save_csv_results(self, results, path):
        """Save per-class results as CSV"""
        
        import csv
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Class', 'IoU', 'Dice', 'Precision', 'Recall', 'F1'])
            
            # Per-class data
            for i, class_name in enumerate(self.class_names):
                writer.writerow([
                    class_name,
                    f"{results['iou_per_class'][i]:.6f}",
                    f"{results['dice_per_class'][i]:.6f}",
                    f"{results['precision_per_class'][i]:.6f}",
                    f"{results['recall_per_class'][i]:.6f}",
                    f"{results['f1_per_class'][i]:.6f}"
                ])
            
            # Overall metrics
            writer.writerow([])
            writer.writerow(['Overall Metrics', '', '', '', '', ''])
            writer.writerow(['Mean IoU', f"{results['miou']:.6f}", '', '', '', ''])
            writer.writerow(['Mean Dice', f"{results['mean_dice']:.6f}", '', '', '', ''])
            writer.writerow(['Pixel Accuracy', f"{results['pixel_accuracy']:.6f}", '', '', '', ''])
    
    def print_summary(self, results):
        """Print evaluation summary to console"""
        
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        print(f"\nOverall Metrics:")
        print(f"  Mean IoU:          {results['miou']:.4f}")
        print(f"  Mean Dice:         {results['mean_dice']:.4f}")
        print(f"  Pixel Accuracy:    {results['pixel_accuracy']:.4f}")
        print(f"  Mean Precision:    {results['mean_precision']:.4f}")
        print(f"  Mean Recall:       {results['mean_recall']:.4f}")
        
        print(f"\nInference Performance:")
        print(f"  Time per image:    {results['inference_time_mean']:.4f} ± {results['inference_time_std']:.4f} sec")
        print(f"  Throughput:        {results['inference_fps']:.2f} FPS")
        
        print(f"\nPer-Class IoU:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name:<20} {results['iou_per_class'][i]:.4f}")
        
        print("\n" + "="*70 + "\n")


def create_dataloader(config, split='test'):
    """
    Create dataloader for evaluation
    
    Args:
        config: Configuration dictionary
        split: Dataset split ('test' or 'val')
    
    Returns:
        DataLoader
    """
    from src.datasets.loveda_dataset import LoveDADataset
    
    dataset = LoveDADataset(
        root_dir=config['data']['root_dir'],
        split=split,
        img_size=config['data']['img_size'],
        augmentation=False  # No augmentation for evaluation
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config['evaluation'].get('batch_size', 8),
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    return loader


def load_model(checkpoint_path, config, device='cuda'):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    print(f"\nLoading model from: {checkpoint_path}")
    
    # Create model
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
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"  Checkpoint from epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"  Validation mIoU: {checkpoint['metrics'].get('miou', 'N/A')}")
    
    print("✓ Model loaded successfully!\n")
    
    return model


def main(args):
    """Main evaluation function"""
    
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Split: {args.split}")
    print(f"Output: {args.output_dir}")
    print("="*70 + "\n")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    
    # Create dataloader
    print(f"Loading {args.split} dataset...")
    test_loader = create_dataloader(config, split=args.split)
    print(f"✓ Dataset loaded: {len(test_loader.dataset)} samples\n")
    
    # Define class names (LoveDA dataset)
    class_names = [
        'Background',
        'Building',
        'Road',
        'Water',
        'Barren',
        'Forest',
        'Agricultural'
    ]
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        num_classes=config['model']['num_classes'],
        class_names=class_names,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        save_predictions=args.save_predictions,
        num_visualizations=args.num_visualizations
    )
    
    # Print summary
    evaluator.print_summary(results)
    
    print(f"✓ Evaluation complete!")
    print(f"✓ Results saved to: {args.output_dir}")
    print(f"\nGenerated files:")
    print(f"  - Metrics: {args.output_dir}/metrics/")
    print(f"  - Figures: {args.output_dir}/figures/")
    if args.save_predictions:
        print(f"  - Predictions: {args.output_dir}/predictions/")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained segmentation model')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate on'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--save_predictions',
        action='store_true',
        help='Save prediction visualizations'
    )
    parser.add_argument(
        '--num_visualizations',
        type=int,
        default=10,
        help='Number of samples to visualize'
    )
    
    args = parser.parse_args()
    main(args)