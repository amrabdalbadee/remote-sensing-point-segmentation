Point-Supervised Remote Sensing Image Segmentation
A comprehensive implementation of partial cross-entropy loss for point-supervised semantic segmentation on remote sensing imagery.

📋 Project Overview
This project implements weakly-supervised semantic segmentation using point annotations instead of full pixel-wise labels. Point annotations are much faster to create, making this approach practical for large-scale remote sensing applications.

Key Features
✅ Partial Cross-Entropy Loss implementation for sparse supervision
✅ Multiple Point Sampling Strategies: Random, Centroid, Boundary-aware, Grid
✅ Modular Architecture: Clean, extensible codebase
✅ U-Net Implementation with customizable depth
✅ Comprehensive Metrics: IoU, Dice, Pixel Accuracy
✅ Experiment Tracking: Automated logging and visualization
✅ Ready for Remote Sensing Datasets: LoveDA, DeepGlobe, etc.
🚀 Quick Start
1. Installation
bash
# Clone the repository
git clone <your-repo-url>
cd remote-sensing-point-segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
2. Download Dataset
bash
# Example for LoveDA dataset
bash scripts/download_data.sh
Or manually download from:

LoveDA: https://github.com/Junjue-Wang/LoveDA
DeepGlobe: https://www.kaggle.com/balraj98/deepglobe-land-cover-classification-dataset
3. Prepare Data
bash
python scripts/preprocess_data.py --dataset loveda --data_dir ./data/raw
4. Train Model
bash
# Train with default configuration
python scripts/train.py --config config/default.yaml

# Resume from checkpoint
python scripts/train.py --config config/default.yaml --resume experiments/baseline/checkpoints/best_model.pth
5. Evaluate Model
bash
python scripts/evaluate.py --checkpoint experiments/baseline/checkpoints/best_model.pth --config config/default.yaml
📁 Project Structure
remote-sensing-point-segmentation/
├── config/                      # Configuration files
│   ├── default.yaml            # Default hyperparameters
│   └── experiment_configs/     # Experiment-specific configs
├── data/                       # Dataset directory
├── src/                        # Source code
│   ├── losses.py              # ✅ Partial CE Loss
│   ├── point_sampling.py      # ✅ Point sampling strategies
│   ├── models/                # Network architectures
│   ├── datasets/              # Dataset loaders
│   ├── utils/                 # Utilities (metrics, visualization)
│   └── training/              # Training logic
├── scripts/                   # Executable scripts
├── experiments/               # Experiment outputs
├── notebooks/                 # Jupyter notebooks
└── tests/                     # Unit tests
🔬 Experiments
Experiment 1: Effect of Number of Points
Hypothesis: Increasing the number of point annotations improves performance with diminishing returns.

bash
# Run with different numbers of points
python scripts/run_experiments.py --experiment num_points
Variables:

Number of points per class: [1, 5, 10, 20, 50, 100]
Baseline: Full supervision
Experiment 2: Sampling Strategy Comparison
Hypothesis: Strategic point placement (centroid, boundary) outperforms random sampling.

bash
# Compare sampling strategies
python scripts/run_experiments.py --experiment sampling_strategy
Strategies:

Random sampling
Centroid-based sampling
Boundary-aware sampling
Grid-based sampling
📊 Results
Results will be automatically saved to experiments/<experiment_name>/results/:

Training curves (loss, IoU)
Confusion matrices
Visualization of predictions
Quantitative metrics (CSV/JSON)
🧪 Usage Examples
Using Different Point Sampling Strategies
python
from src.point_sampling import create_point_sampler

# Random sampling
sampler = create_point_sampler('random', num_points_per_class=5)

# Centroid-based sampling
sampler = create_point_sampler('centroid', num_points_per_class=5, min_object_size=10)

# Boundary-aware sampling
sampler = create_point_sampler('boundary', num_points_per_class=5, boundary_thickness=5)

# Grid sampling
sampler = create_point_sampler('grid', grid_size=32)

# Sample points from a mask
point_labels, label_mask = sampler.sample(segmentation_mask)
Using Partial Cross-Entropy Loss
python
from src.losses import PartialCrossEntropyLoss

# Create loss function
criterion = PartialCrossEntropyLoss(ignore_index=255)

# Compute loss
loss = criterion(predictions, point_labels, label_mask)
Custom Training Loop
python
from src.training.trainer import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    point_sampler=sampler,
    num_classes=7,
    device='cuda'
)

history = trainer.train(num_epochs=100)
📈 Visualization
Explore results interactively:

bash
jupyter notebook notebooks/03_results_analysis.ipynb
🧪 Testing
Run unit tests:

bash
pytest tests/
📝 Configuration
Edit config/default.yaml to customize:

Model architecture (U-Net variants)
Point sampling strategy and parameters
Training hyperparameters
Data augmentation settings
🎯 Key Implementation Details
Partial Cross-Entropy Loss
The loss only computes gradients for labeled pixels:

L = (1/N) Σ mask_i * CE(pred_i, target_i)
where N is the number of labeled pixels.

Point Sampling
Four strategies are implemented:

Random: Uniform random sampling from each class
Centroid: Samples from centers of connected components
Boundary: Samples near object boundaries
Grid: Regular grid sampling
Evaluation
Models are evaluated using:

Mean Intersection over Union (mIoU)
Per-class IoU
Dice coefficient
Pixel accuracy
Precision and Recall
Importantly, evaluation is always on full masks, even when training with points.

🐛 Troubleshooting
Out of Memory (OOM)
Reduce batch size in config/default.yaml
Use model: light_unet for a smaller model
Reduce image size
Slow Training
Increase num_workers in config
Use mixed precision training (add to code)
Use a smaller model variant
Poor Performance
Try more point annotations
Experiment with different sampling strategies
Increase training epochs
Add data augmentation
📚 References
U-Net: Ronneberger et al., 2015
LoveDA Dataset: Wang et al., 2021
Point Supervision: Various weakly-supervised learning papers
📄 License
MIT License

🤝 Contributing
Contributions are welcome! Please:

Fork the repository
Create a feature branch
Submit a pull request
📧 Contact
For questions or issues, please open a GitHub issue.

Note: Make sure to update the dataset paths and configuration files based on your specific setup before running experiments.

