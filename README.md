# Remote Sensing Point Segmentation

A deep learning framework for point cloud segmentation in remote sensing applications, featuring advanced sampling strategies and partial cross-entropy loss implementation.

## Project Structure

```
remote-sensing-point-segmentation/
├── config/                     # Configuration files
│   ├── default.yaml            # Default configuration settings
│   └── experiment_configs/     # Experiment-specific configurations
├── data/                       # Dataset storage directory
├── src/                        # Source code
│   ├── losses.py               # Partial CE Loss implementation
│   ├── point_sampling.py       # Point cloud sampling strategies
│   ├── models/                 # Neural network model architectures
│   ├── datasets/               # Dataset loaders and processors
│   ├── utils/                  # Utility functions and helpers
│   └── training/               # Training loop and related scripts
├── scripts/                    # Executable scripts for running experiments
├── experiments/                # Experiment results and logs
├── notebooks/                  # Jupyter notebooks for analysis
└── tests/                      # Unit tests and integration tests
```

## Features

- **Partial Cross-Entropy Loss**: Custom loss function for handling partially labeled data
- **Advanced Sampling Strategies**: Multiple point cloud sampling techniques for efficient processing
- **Modular Architecture**: Easy to extend and customize for different tasks
- **Experiment Tracking**: Built-in support for experiment management and logging
- **Configuration Management**: YAML-based configuration system for reproducible experiments

## Installation

```bash
# Clone the repository
git clone https://github.com/amrabdalbadee/remote-sensing-point-segmentation.git
cd remote-sensing-point-segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Train with default configuration
python scripts/train.py --config config/default.yaml

# Train with custom experiment configuration
python scripts/train.py --config config/experiment_configs/experiment_01.yaml
```

### Evaluation

```bash
# Evaluate a trained model
python scripts/evaluate.py --model_path experiments/model_checkpoint.pth --config config/default.yaml
```

### Inference

```bash
# Run inference on new data
python scripts/inference.py --input data/test_samples/ --model_path experiments/model_checkpoint.pth
```

## Configuration

Edit `config/default.yaml` to customize:

- Model architecture parameters
- Training hyperparameters (learning rate, batch size, epochs)
- Data augmentation settings
- Loss function parameters
- Sampling strategy configuration

Example configuration:

```yaml
model:
  name: PointNet++
  num_classes: 10
  
training:
  batch_size: 16
  learning_rate: 0.001
  epochs: 100
  
loss:
  type: partial_ce
  weight_unlabeled: 0.1
```

## Dataset Preparation

Place your point cloud data in the `data/` directory with the following structure:

```
data/
├── train/
│   ├── points/
│   └── labels/
├── val/
│   ├── points/
│   └── labels/
└── test/
    └── points/
```

## Key Components

### Partial Cross-Entropy Loss (`src/losses.py`)

Implements a custom loss function that handles partially labeled point clouds, allowing the model to learn from both labeled and unlabeled points.

### Point Sampling (`src/point_sampling.py`)

Provides various sampling strategies including:
- Random sampling
- Farthest point sampling
- Grid-based sampling
- Density-aware sampling

## Results

Track your experiment results in the `experiments/` directory. Each experiment creates:
- Model checkpoints
- Training logs
- Evaluation metrics
- Visualization outputs

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{remote_sensing_segmentation,
  title={Remote Sensing Point Segmentation Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/remote-sensing-point-segmentation}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact [your.email@example.com]

## Acknowledgments

- Thanks to the open-source community for inspiration and tools
- Built with PyTorch and other amazing libraries