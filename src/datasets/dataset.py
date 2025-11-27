"""
LoveDA Dataset Loader for Remote Sensing Semantic Segmentation

LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation
Paper: https://arxiv.org/abs/2110.08733
Dataset: https://github.com/Junjue-Wang/LoveDA

Classes:
0: Background
1: Building
2: Road
3: Water
4: Barren
5: Forest
6: Agricultural
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import Optional, Tuple


class LoveDADataset(Dataset):
    """
    LoveDA Dataset for semantic segmentation.
    
    Expected directory structure:
    root_dir/
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
    
    # Class definitions
    CLASSES = [
        'background', 'building', 'road', 'water', 
        'barren', 'forest', 'agricultural'
    ]
    
    # Color palette for visualization (RGB)
    PALETTE = [
        [0, 0, 0],        # Background - Black
        [255, 0, 0],      # Building - Red
        [255, 255, 0],    # Road - Yellow
        [0, 0, 255],      # Water - Blue
        [159, 129, 183],  # Barren - Purple
        [0, 255, 0],      # Forest - Green
        [255, 195, 128],  # Agricultural - Orange
    ]
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'Train',
        domain: str = 'both',  # 'Urban', 'Rural', or 'both'
        img_size: int = 256,
        augmentation: bool = True,
        normalize: bool = True,
        ignore_index: int = 255
    ):
        """
        Args:
            root_dir: Path to LoveDA dataset root
            split: 'Train' or 'Val'
            domain: 'Urban', 'Rural', or 'both'
            img_size: Size to resize images
            augmentation: Apply data augmentation (only for training)
            normalize: Apply ImageNet normalization
            ignore_index: Value for pixels to ignore in loss
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.domain = domain
        self.img_size = img_size
        self.augmentation = augmentation and (split == 'Train')
        self.normalize = normalize
        self.ignore_index = ignore_index
        
        # Collect image paths
        self.samples = self._load_samples()
        
        # Setup transforms
        self.transform = self._get_transforms()
        
        print(f"Loaded {len(self.samples)} samples from LoveDA {split} set ({domain})")
    
    def _load_samples(self):
        """Load all image and mask paths"""
        samples = []
        
        domains = ['Urban', 'Rural'] if self.domain == 'both' else [self.domain]
        
        for domain in domains:
            img_dir = self.root_dir / self.split / domain / 'images_png'
            mask_dir = self.root_dir / self.split / domain / 'masks_png'
            
            if not img_dir.exists():
                print(f"Warning: {img_dir} does not exist")
                continue
            
            # Get all image files
            img_files = sorted(list(img_dir.glob('*.png')))
            
            for img_path in img_files:
                mask_path = mask_dir / img_path.name
                
                if mask_path.exists():
                    samples.append({
                        'image': str(img_path),
                        'mask': str(mask_path),
                        'domain': domain
                    })
        
        return samples
    
    def _get_transforms(self):
        """Get augmentation and normalization transforms"""
        transform_list = []
        
        # Resize
        transform_list.append(A.Resize(self.img_size, self.img_size))
        
        # Augmentation (only for training)
        if self.augmentation:
            transform_list.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=45,
                    p=0.5
                ),
                A.OneOf([
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                    A.HueSaturationValue(p=1),
                ], p=0.3),
                A.OneOf([
                    A.GaussianBlur(p=1),
                    A.GaussNoise(p=1),
                ], p=0.2),
            ])
        
        # Normalization
        if self.normalize:
            transform_list.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        # Convert to tensor
        transform_list.append(ToTensorV2())
        
        return A.Compose(transform_list)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor of shape [3, H, W]
            mask: Tensor of shape [H, W] with class indices
        """
        sample = self.samples[idx]
        
        # Load image and mask
        image = np.array(Image.open(sample['image']).convert('RGB'))
        mask = np.array(Image.open(sample['mask']))
        
        # Convert mask to class indices (0-6)
        # LoveDA masks are RGB images, need conversion
        mask = self._mask_to_class(mask)
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        # ToTensorV2 already converts mask to tensor, so check type
        mask = transformed['mask']
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()
        
        return image, mask
    
    def _mask_to_class(self, mask):
        """
        Convert RGB mask to class indices.
        
        Args:
            mask: RGB mask of shape [H, W, 3]
        
        Returns:
            class_mask: Class indices of shape [H, W]
        """
        if len(mask.shape) == 2:
            # Already in class format
            return mask
        
        h, w = mask.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Convert RGB to class index
        for class_idx, color in enumerate(self.PALETTE):
            # Find pixels matching this color
            matches = np.all(mask == color, axis=-1)
            class_mask[matches] = class_idx
        
        return class_mask
    
    @classmethod
    def get_class_names(cls):
        """Return list of class names"""
        return cls.CLASSES
    
    @classmethod
    def get_palette(cls):
        """Return color palette"""
        return cls.PALETTE
    
    def visualize_sample(self, idx, save_path=None):
        """Visualize a sample with image and mask"""
        import matplotlib.pyplot as plt
        
        sample = self.samples[idx]
        image = np.array(Image.open(sample['image']))
        mask = np.array(Image.open(sample['mask']))
        mask = self._mask_to_class(mask)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(image)
        axes[0].set_title('Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='tab10')
        axes[1].set_title('Mask')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()


class DeepGlobeDataset(Dataset):
    """
    DeepGlobe Land Cover Classification Dataset
    
    Classes:
    0: Urban land
    1: Agriculture land
    2: Rangeland
    3: Forest land
    4: Water
    5: Barren land
    6: Unknown
    """
    
    CLASSES = [
        'urban', 'agriculture', 'rangeland', 
        'forest', 'water', 'barren', 'unknown'
    ]
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        img_size: int = 256,
        augmentation: bool = True,
        normalize: bool = True,
        ignore_index: int = 255
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.augmentation = augmentation and (split == 'train')
        self.normalize = normalize
        self.ignore_index = ignore_index
        
        # Load samples
        self.samples = self._load_samples()
        
        # Setup transforms
        self.transform = self._get_transforms()
        
        print(f"Loaded {len(self.samples)} samples from DeepGlobe {split} set")
    
    def _load_samples(self):
        """Load image and mask paths"""
        samples = []
        
        img_dir = self.root_dir / self.split / 'images'
        mask_dir = self.root_dir / self.split / 'masks'
        
        if not img_dir.exists():
            raise ValueError(f"Image directory not found: {img_dir}")
        
        for img_path in sorted(img_dir.glob('*.jpg')):
            # Corresponding mask (usually has _mask suffix)
            mask_name = img_path.stem + '_mask.png'
            mask_path = mask_dir / mask_name
            
            if mask_path.exists():
                samples.append({
                    'image': str(img_path),
                    'mask': str(mask_path)
                })
        
        return samples
    
    def _get_transforms(self):
        """Get transforms"""
        transform_list = [A.Resize(self.img_size, self.img_size)]
        
        if self.augmentation:
            transform_list.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
            ])
        
        if self.normalize:
            transform_list.append(
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        
        transform_list.append(ToTensorV2())
        
        return A.Compose(transform_list)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = np.array(Image.open(sample['image']).convert('RGB'))
        mask = np.array(Image.open(sample['mask']))
        
        # Ensure mask is single channel
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        # ToTensorV2 already converts mask to tensor, so check type
        mask = transformed['mask']
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()
        
        return image, mask
    
    @classmethod
    def get_class_names(cls):
        return cls.CLASSES


def create_dataset(dataset_name: str, **kwargs):
    """
    Factory function to create datasets.
    
    Args:
        dataset_name: 'loveda' or 'deepglobe'
        **kwargs: Arguments for the dataset
    
    Returns:
        Dataset instance
    """
    datasets = {
        'loveda': LoveDADataset,
        'deepglobe': DeepGlobeDataset,
    }
    
    if dataset_name.lower() not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Choose from {list(datasets.keys())}")
    
    return datasets[dataset_name.lower()](**kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test LoveDA dataset
    dataset = LoveDADataset(
        root_dir='./data/LoveDA',
        split='Train',
        domain='both',
        img_size=256,
        augmentation=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.get_class_names()}")
    
    # Test loading a sample
    if len(dataset) > 0:
        image, mask = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask unique values: {torch.unique(mask)}")