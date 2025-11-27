"""
Data Preprocessing Script for Remote Sensing Datasets

This script handles:
1. Dataset download instructions
2. Data format conversion
3. Train/val/test splits
4. Data statistics and visualization

Usage:
    python scripts/preprocess_data.py --dataset loveda --data_dir ./data/raw
"""
import os
import sys
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))


class DatasetPreprocessor:
    """Base class for dataset preprocessing"""
    
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process(self):
        """Main processing pipeline"""
        raise NotImplementedError


class LoveDAPreprocessor(DatasetPreprocessor):
    """
    Preprocessor for LoveDA dataset.
    
    Expected input structure:
    data_dir/
        Train/
            Urban/
                images_png/
                masks_png/
            Rural/
                images_png/
                masks_png/
        Val/
            Urban/
                images_png/
                masks_png/
            Rural/
                images_png/
                masks_png/
    """
    
    def __init__(self, data_dir, output_dir):
        super().__init__(data_dir, output_dir)
        self.class_names = [
            'background', 'building', 'road', 'water',
            'barren', 'forest', 'agricultural'
        ]
    
    def process(self):
        """Process LoveDA dataset"""
        print("\n" + "="*60)
        print("Processing LoveDA Dataset")
        print("="*60)
        
        # Check if data exists
        if not self.data_dir.exists():
            print(f"Error: Data directory not found: {self.data_dir}")
            print("\nTo download LoveDA dataset:")
            print("1. Visit: https://github.com/Junjue-Wang/LoveDA")
            print("2. Download the dataset")
            print("3. Extract to:", self.data_dir)
            return False
        
        # Verify structure
        print("\nVerifying dataset structure...")
        splits = ['Train', 'Val']
        domains = ['Urban', 'Rural']
        
        for split in splits:
            for domain in domains:
                img_dir = self.data_dir / split / domain / 'images_png'
                mask_dir = self.data_dir / split / domain / 'masks_png'
                
                if not img_dir.exists() or not mask_dir.exists():
                    print(f"Warning: Missing {split}/{domain} data")
                else:
                    num_images = len(list(img_dir.glob('*.png')))
                    num_masks = len(list(mask_dir.glob('*.png')))
                    print(f"✓ {split}/{domain}: {num_images} images, {num_masks} masks")
        
        # Compute dataset statistics
        print("\nComputing dataset statistics...")
        self.compute_statistics()
        
        # Visualize samples
        print("\nGenerating sample visualizations...")
        self.visualize_samples()
        
        print("\n✓ LoveDA dataset preprocessing complete!")
        return True
    
    def compute_statistics(self):
        """Compute and save dataset statistics"""
        stats = {
            'splits': {},
            'class_distribution': {name: 0 for name in self.class_names}
        }
        
        splits = ['Train', 'Val']
        
        for split in splits:
            split_stats = {'total_samples': 0, 'domains': {}}
            
            for domain in ['Urban', 'Rural']:
                img_dir = self.data_dir / split / domain / 'images_png'
                mask_dir = self.data_dir / split / domain / 'masks_png'
                
                if img_dir.exists():
                    images = list(img_dir.glob('*.png'))
                    split_stats['domains'][domain] = len(images)
                    split_stats['total_samples'] += len(images)
                    
                    # Sample masks for class distribution
                    if mask_dir.exists():
                        sample_masks = list(mask_dir.glob('*.png'))[:10]
                        for mask_path in sample_masks:
                            mask = np.array(Image.open(mask_path))
                            unique, counts = np.unique(mask, return_counts=True)
                            for cls, count in zip(unique, counts):
                                if cls < len(self.class_names):
                                    stats['class_distribution'][self.class_names[cls]] += count
            
            stats['splits'][split] = split_stats
        
        # Save statistics
        stats_file = self.output_dir / 'dataset_statistics.txt'
        with open(stats_file, 'w') as f:
            f.write("LoveDA Dataset Statistics\n")
            f.write("="*60 + "\n\n")
            
            for split, split_stats in stats['splits'].items():
                f.write(f"{split} Set:\n")
                f.write(f"  Total samples: {split_stats['total_samples']}\n")
                for domain, count in split_stats['domains'].items():
                    f.write(f"  {domain}: {count}\n")
                f.write("\n")
            
            f.write("Class Distribution (sampled):\n")
            total_pixels = sum(stats['class_distribution'].values())
            for class_name, count in stats['class_distribution'].items():
                percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
                f.write(f"  {class_name}: {percentage:.2f}%\n")
        
        print(f"Statistics saved to {stats_file}")
    
    def visualize_samples(self, num_samples=6):
        """Visualize random samples from the dataset"""
        vis_dir = self.output_dir / 'sample_visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        # Sample from train set
        img_dir = self.data_dir / 'Train' / 'Urban' / 'images_png'
        mask_dir = self.data_dir / 'Train' / 'Urban' / 'masks_png'
        
        if not img_dir.exists():
            return
        
        images = list(img_dir.glob('*.png'))
        sampled = np.random.choice(images, min(num_samples, len(images)), replace=False)
        
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3*num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, img_path in enumerate(sampled):
            mask_path = mask_dir / img_path.name
            
            image = Image.open(img_path)
            mask = Image.open(mask_path)
            
            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f'Image: {img_path.name}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask, cmap='tab10')
            axes[i, 1].set_title('Mask')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(vis_dir / 'sample_images.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Sample visualizations saved to {vis_dir}")


class DeepGlobePreprocessor(DatasetPreprocessor):
    """Preprocessor for DeepGlobe dataset"""
    
    def __init__(self, data_dir, output_dir):
        super().__init__(data_dir, output_dir)
        self.class_names = [
            'urban', 'agriculture', 'rangeland',
            'forest', 'water', 'barren', 'unknown'
        ]
    
    def process(self):
        """Process DeepGlobe dataset"""
        print("\n" + "="*60)
        print("Processing DeepGlobe Dataset")
        print("="*60)
        
        if not self.data_dir.exists():
            print(f"Error: Data directory not found: {self.data_dir}")
            print("\nTo download DeepGlobe dataset:")
            print("1. Visit: https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset")
            print("2. Download the dataset")
            print("3. Extract to:", self.data_dir)
            return False
        
        print("\nDeepGlobe preprocessing complete!")
        print("Note: Implement specific preprocessing as needed for your data format")
        return True


def create_train_val_split(data_dir, output_dir, val_ratio=0.2, seed=42):
    """
    Create train/val split if not already split.
    
    Args:
        data_dir: Directory containing all data
        output_dir: Output directory for splits
        val_ratio: Fraction of data for validation
        seed: Random seed
    """
    print("\n" + "="*60)
    print("Creating Train/Val Split")
    print("="*60)
    
    np.random.seed(seed)
    
    # This is a template - adapt based on your data structure
    print(f"Validation ratio: {val_ratio}")
    print(f"Random seed: {seed}")
    print("\nNote: Implement split logic based on your dataset structure")


def download_instructions(dataset_name):
    """Print download instructions for datasets"""
    print("\n" + "="*60)
    print(f"Download Instructions for {dataset_name.upper()}")
    print("="*60)
    
    if dataset_name.lower() == 'loveda':
        print("\nLoveDA Dataset:")
        print("1. Visit: https://github.com/Junjue-Wang/LoveDA")
        print("2. Download from one of the provided links:")
        print("   - Google Drive")
        print("   - Baidu Netdisk")
        print("3. Extract the downloaded file")
        print("4. Expected structure:")
        print("   LoveDA/")
        print("     Train/")
        print("       Urban/images_png/")
        print("       Urban/masks_png/")
        print("       Rural/images_png/")
        print("       Rural/masks_png/")
        print("     Val/")
        print("       Urban/images_png/")
        print("       Urban/masks_png/")
        print("       Rural/images_png/")
        print("       Rural/masks_png/")
    
    elif dataset_name.lower() == 'deepglobe':
        print("\nDeepGlobe Dataset:")
        print("1. Visit: https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset")
        print("2. Download the dataset (requires Kaggle account)")
        print("3. Extract the downloaded file")
    
    else:
        print(f"\nNo download instructions available for {dataset_name}")
    
    print("\n" + "="*60)


def main(args):
    """Main preprocessing function"""
    
    print("\n" + "="*70)
    print("DATA PREPROCESSING")
    print("="*70)
    
    # Show download instructions if requested
    if args.download_instructions:
        download_instructions(args.dataset)
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process dataset
    if args.dataset.lower() == 'loveda':
        preprocessor = LoveDAPreprocessor(args.data_dir, output_dir)
    elif args.dataset.lower() == 'deepglobe':
        preprocessor = DeepGlobePreprocessor(args.data_dir, output_dir)
    else:
        print(f"Unknown dataset: {args.dataset}")
        print("Supported datasets: loveda, deepglobe")
        return
    
    success = preprocessor.process()
    
    if success:
        print("\n" + "="*70)
        print("PREPROCESSING COMPLETE!")
        print("="*70)
        print(f"Processed data information saved to: {output_dir}")
        print("\nNext steps:")
        print("1. Update config/default.yaml with the correct data path")
        print("2. Run: python scripts/train.py --config config/default.yaml")
        print("="*70 + "\n")
    else:
        print("\nPreprocessing failed. Please check the error messages above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess remote sensing datasets')
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['loveda', 'deepglobe'],
        help='Dataset name'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/raw',
        help='Directory containing raw dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/processed',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--download_instructions',
        action='store_true',
        help='Show download instructions for the dataset'
    )
    
    args = parser.parse_args()
    main(args)