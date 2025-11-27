import numpy as np
import torch
from scipy.ndimage import label as scipy_label
from scipy.ndimage import center_of_mass, distance_transform_edt
from typing import Tuple, Optional

###Point Sampling Strategies for Simulating Sparse Annotations

class PointSampler:
    """Base class for point sampling strategies"""
    
    def __init__(self, num_points_per_class=5, ignore_index=255):
        """
        Args:
            num_points_per_class (int): Number of points to sample per class
            ignore_index (int): Value for unlabeled pixels
        """
        self.num_points_per_class = num_points_per_class
        self.ignore_index = ignore_index
    
    def sample(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample points from a segmentation mask.
        
        Args:
            mask (np.ndarray): [H, W] segmentation mask
        
        Returns:
            point_labels (np.ndarray): [H, W] sparse label map
            label_mask (np.ndarray): [H, W] binary mask (1=labeled, 0=unlabeled)
        """
        raise NotImplementedError


class RandomPointSampler(PointSampler):
    """Randomly samples points from each class"""
    
    def sample(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly sample points from each class in the mask.
        
        Args:
            mask (np.ndarray): [H, W] segmentation mask
        
        Returns:
            point_labels (np.ndarray): [H, W] sparse labels
            label_mask (np.ndarray): [H, W] binary mask
        """
        h, w = mask.shape
        point_labels = np.full((h, w), self.ignore_index, dtype=np.int64)
        label_mask = np.zeros((h, w), dtype=np.float32)
        
        # Get unique classes (excluding ignore_index)
        unique_classes = np.unique(mask)
        unique_classes = unique_classes[unique_classes != self.ignore_index]
        
        for class_id in unique_classes:
            # Find all pixels of this class
            class_pixels = np.argwhere(mask == class_id)
            
            if len(class_pixels) == 0:
                continue
            
            # Sample random points
            num_samples = min(self.num_points_per_class, len(class_pixels))
            sampled_indices = np.random.choice(
                len(class_pixels), 
                size=num_samples, 
                replace=False
            )
            sampled_points = class_pixels[sampled_indices]
            
            # Mark sampled points
            for point in sampled_points:
                y, x = point
                point_labels[y, x] = class_id
                label_mask[y, x] = 1.0
        
        return point_labels, label_mask


class CentroidPointSampler(PointSampler):
    """Samples points from object centroids (centers of connected components)"""
    
    def __init__(self, num_points_per_class=5, ignore_index=255, 
                 min_object_size=10):
        super().__init__(num_points_per_class, ignore_index)
        self.min_object_size = min_object_size
    
    def sample(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample points from centroids of connected components.
        
        Args:
            mask (np.ndarray): [H, W] segmentation mask
        
        Returns:
            point_labels (np.ndarray): [H, W] sparse labels
            label_mask (np.ndarray): [H, W] binary mask
        """
        h, w = mask.shape
        point_labels = np.full((h, w), self.ignore_index, dtype=np.int64)
        label_mask = np.zeros((h, w), dtype=np.float32)
        
        unique_classes = np.unique(mask)
        unique_classes = unique_classes[unique_classes != self.ignore_index]
        
        for class_id in unique_classes:
            # Create binary mask for this class
            class_mask = (mask == class_id).astype(np.int32)
            
            # Find connected components
            labeled_components, num_components = scipy_label(class_mask)
            
            # Get centroids of each component
            centroids = []
            for comp_id in range(1, num_components + 1):
                component_mask = (labeled_components == comp_id)
                size = component_mask.sum()
                
                # Skip small objects
                if size < self.min_object_size:
                    continue
                
                # Compute centroid
                centroid = center_of_mass(component_mask)
                centroids.append((int(centroid[0]), int(centroid[1])))
            
            # Sample from available centroids
            if len(centroids) == 0:
                continue
            
            num_samples = min(self.num_points_per_class, len(centroids))
            sampled_centroids = np.random.choice(
                len(centroids), 
                size=num_samples, 
                replace=False
            )
            
            for idx in sampled_centroids:
                y, x = centroids[idx]
                # Ensure within bounds
                y = np.clip(y, 0, h - 1)
                x = np.clip(x, 0, w - 1)
                point_labels[y, x] = class_id
                label_mask[y, x] = 1.0
        
        return point_labels, label_mask


class BoundaryAwarePointSampler(PointSampler):
    """Samples points near object boundaries"""
    
    def __init__(self, num_points_per_class=5, ignore_index=255, 
                 boundary_thickness=5):
        super().__init__(num_points_per_class, ignore_index)
        self.boundary_thickness = boundary_thickness
    
    def sample(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample points near object boundaries.
        
        Args:
            mask (np.ndarray): [H, W] segmentation mask
        
        Returns:
            point_labels (np.ndarray): [H, W] sparse labels
            label_mask (np.ndarray): [H, W] binary mask
        """
        h, w = mask.shape
        point_labels = np.full((h, w), self.ignore_index, dtype=np.int64)
        label_mask = np.zeros((h, w), dtype=np.float32)
        
        unique_classes = np.unique(mask)
        unique_classes = unique_classes[unique_classes != self.ignore_index]
        
        for class_id in unique_classes:
            class_mask = (mask == class_id).astype(np.uint8)
            
            # Compute distance transform
            dist_transform = distance_transform_edt(class_mask)
            
            # Find pixels near boundary (within boundary_thickness)
            boundary_pixels = np.argwhere(
                (dist_transform > 0) & 
                (dist_transform <= self.boundary_thickness)
            )
            
            if len(boundary_pixels) == 0:
                # Fallback to all class pixels if no boundary found
                boundary_pixels = np.argwhere(class_mask > 0)
            
            if len(boundary_pixels) == 0:
                continue
            
            # Sample points
            num_samples = min(self.num_points_per_class, len(boundary_pixels))
            sampled_indices = np.random.choice(
                len(boundary_pixels), 
                size=num_samples, 
                replace=False
            )
            sampled_points = boundary_pixels[sampled_indices]
            
            for point in sampled_points:
                y, x = point
                point_labels[y, x] = class_id
                label_mask[y, x] = 1.0
        
        return point_labels, label_mask


class GridPointSampler(PointSampler):
    """Samples points on a regular grid within each class"""
    
    def __init__(self, grid_size=32, ignore_index=255, num_points_per_class=None):
        """
        Args:
            grid_size (int): Spacing between grid points in pixels
            ignore_index (int): Value for unlabeled pixels
            num_points_per_class (int, optional): Ignored for grid sampler (kept for compatibility)
        """
        super().__init__(num_points_per_class=0, ignore_index=ignore_index)
        self.grid_size = grid_size
    
    def sample(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample points on a grid within each class region.
        
        Args:
            mask (np.ndarray): [H, W] segmentation mask
        
        Returns:
            point_labels (np.ndarray): [H, W] sparse labels
            label_mask (np.ndarray): [H, W] binary mask
        """
        h, w = mask.shape
        point_labels = np.full((h, w), self.ignore_index, dtype=np.int64)
        label_mask = np.zeros((h, w), dtype=np.float32)
        
        # Create grid coordinates
        grid_y = np.arange(0, h, self.grid_size)
        grid_x = np.arange(0, w, self.grid_size)
        
        for y in grid_y:
            for x in grid_x:
                if y < h and x < w:
                    class_id = mask[y, x]
                    if class_id != self.ignore_index:
                        point_labels[y, x] = class_id
                        label_mask[y, x] = 1.0
        
        return point_labels, label_mask


def create_point_sampler(strategy='random', **kwargs):
    """
    Factory function to create point samplers.
    
    Args:
        strategy (str): 'random', 'centroid', 'boundary', or 'grid'
        **kwargs: Additional arguments for the sampler
    
    Returns:
        PointSampler: Instance of the requested sampler
    """
    samplers = {
        'random': RandomPointSampler,
        'centroid': CentroidPointSampler,
        'boundary': BoundaryAwarePointSampler,
        'grid': GridPointSampler,
    }
    
    if strategy not in samplers:
        raise ValueError(f"Unknown strategy: {strategy}. "
                        f"Choose from {list(samplers.keys())}")
    
    return samplers[strategy](**kwargs)


def batch_sample_points(masks, sampler, device='cpu'):
    """
    Apply point sampling to a batch of masks.
    
    Args:
        masks (torch.Tensor): [B, H, W] batch of segmentation masks
        sampler (PointSampler): Point sampling strategy
        device (str): Device to move tensors to
    
    Returns:
        point_labels (torch.Tensor): [B, H, W] sparse labels
        label_masks (torch.Tensor): [B, H, W] binary masks
    """
    batch_size = masks.shape[0]
    point_labels_list = []
    label_masks_list = []
    
    for i in range(batch_size):
        mask_np = masks[i].cpu().numpy()
        point_labels, label_mask = sampler.sample(mask_np)
        point_labels_list.append(torch.from_numpy(point_labels))
        label_masks_list.append(torch.from_numpy(label_mask))
    
    point_labels_batch = torch.stack(point_labels_list).to(device)
    label_masks_batch = torch.stack(label_masks_list).to(device)
    
    return point_labels_batch, label_masks_batch
