"""
Installation Test Script

Run this to verify your setup is correct.

Usage: python test_installation.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import torchvision
        print(f"✓ TorchVision {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ TorchVision import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import albumentations
        print(f"✓ Albumentations {albumentations.__version__}")
    except ImportError as e:
        print(f"✗ Albumentations import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    return True


def test_custom_modules():
    """Test custom module imports"""
    print("\nTesting custom modules...")
    
    try:
        from src.losses import PartialCrossEntropyLoss
        print("✓ Losses module")
    except ImportError as e:
        print(f"✗ Losses module import failed: {e}")
        return False
    
    try:
        from src.point_sampling import RandomPointSampler, create_point_sampler
        print("✓ Point sampling module")
    except ImportError as e:
        print(f"✗ Point sampling module import failed: {e}")
        return False
    
    try:
        from src.models.unet import UNet
        print("✓ Models module")
    except ImportError as e:
        print(f"✗ Models module import failed: {e}")
        return False
    
    try:
        from src.utils.metrics import SegmentationMetrics
        print("✓ Metrics module")
    except ImportError as e:
        print(f"✗ Metrics module import failed: {e}")
        return False
    
    try:
        from src.utils.visualization import plot_training_history
        print("✓ Visualization module")
    except ImportError as e:
        print(f"✗ Visualization module import failed: {e}")
        return False
    
    return True


def test_model_forward():
    """Test model forward pass"""
    print("\nTesting model forward pass...")
    
    try:
        import torch
        from src.models.unet import UNet
        
        model = UNet(in_channels=3, num_classes=5)
        x = torch.randn(2, 3, 256, 256)
        output = model(x)
        
        assert output.shape == (2, 5, 256, 256), f"Unexpected output shape: {output.shape}"
        print(f"✓ Model forward pass successful! Output shape: {output.shape}")
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model has {num_params:,} parameters")
        
        return True
    except Exception as e:
        print(f"✗ Model forward pass failed: {e}")
        return False


def test_loss_computation():
    """Test loss computation"""
    print("\nTesting loss computation...")
    
    try:
        import torch
        from src.losses import PartialCrossEntropyLoss
        
        criterion = PartialCrossEntropyLoss(ignore_index=255)
        
        # Create dummy data
        predictions = torch.randn(2, 5, 256, 256)
        targets = torch.randint(0, 5, (2, 256, 256))
        label_mask = torch.rand(2, 256, 256) > 0.5
        
        # Compute loss
        loss = criterion(predictions, targets, label_mask.float())
        
        assert loss.item() >= 0, "Loss should be non-negative"
        print(f"✓ Loss computation successful! Loss: {loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        return False


def test_point_sampling():
    """Test point sampling"""
    print("\nTesting point sampling...")
    
    try:
        import numpy as np
        from src.point_sampling import RandomPointSampler, create_point_sampler
        
        # Test random sampler
        sampler = RandomPointSampler(num_points_per_class=5)
        mask = np.random.randint(0, 5, (256, 256))
        
        point_labels, label_mask = sampler.sample(mask)
        
        num_points = (label_mask > 0).sum()
        assert num_points > 0, "Should sample at least some points"
        print(f"✓ Point sampling successful! Sampled {num_points} points")
        
        # Test factory function
        samplers = ['random', 'centroid', 'boundary', 'grid']
        for strategy in samplers:
            sampler = create_point_sampler(strategy, num_points_per_class=5)
            point_labels, label_mask = sampler.sample(mask)
            print(f"✓ {strategy.capitalize()} sampler working")
        
        return True
    except Exception as e:
        print(f"✗ Point sampling failed: {e}")
        return False


def test_metrics():
    """Test metrics computation"""
    print("\nTesting metrics computation...")
    
    try:
        import torch
        from src.utils.metrics import SegmentationMetrics, compute_iou
        
        # Create metrics tracker
        metrics_tracker = SegmentationMetrics(num_classes=5, ignore_index=255)
        
        # Create dummy predictions and targets
        predictions = torch.randint(0, 5, (4, 256, 256))
        targets = torch.randint(0, 5, (4, 256, 256))
        
        # Update metrics
        metrics_tracker.update(predictions, targets)
        
        # Get metrics
        metrics = metrics_tracker.get_metrics()
        
        assert 'miou' in metrics, "Should compute mIoU"
        assert 0 <= metrics['miou'] <= 1, "mIoU should be between 0 and 1"
        print(f"✓ Metrics computation successful! mIoU: {metrics['miou']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Metrics computation failed: {e}")
        return False


def test_cuda_availability():
    """Test CUDA availability"""
    print("\nChecking CUDA availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA is available!")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠ CUDA is not available. Training will use CPU (slower).")
        
        return True
    except Exception as e:
        print(f"✗ CUDA check failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*70)
    print("INSTALLATION TEST")
    print("="*70)
    
    tests = [
        ("Package Imports", test_imports),
        ("Custom Modules", test_custom_modules),
        ("Model Forward Pass", test_model_forward),
        ("Loss Computation", test_loss_computation),
        ("Point Sampling", test_point_sampling),
        ("Metrics Computation", test_metrics),
        ("CUDA Availability", test_cuda_availability),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} encountered an error: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} - {test_name}")
    
    print("="*70)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your installation is ready.")
        print("\nNext steps:")
        print("1. Download dataset: python scripts/preprocess_data.py --dataset loveda --download_instructions")
        print("2. Preprocess data: python scripts/preprocess_data.py --dataset loveda --data_dir ./data/raw")
        print("3. Update config: Edit config/default.yaml with your data path")
        print("4. Start training: python scripts/train.py --config config/default.yaml")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please check the errors above.")
        print("Common fixes:")
        print("- Reinstall packages: pip install -r requirements.txt")
        print("- Check Python version (3.8+ required)")
        print("- Ensure all __init__.py files exist in src/ directories")
    
    print("="*70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)