#!/bin/bash

# Automated Setup Script for Point-Supervised Segmentation Project
# Usage: bash setup.sh

set -e  # Exit on error

echo "=========================================="
echo "Point-Supervised Segmentation Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create project structure
echo ""
echo "Creating project structure..."

# Main directories
mkdir -p config/experiment_configs
mkdir -p data/{raw,processed,splits}
mkdir -p src/{models,datasets,utils,training}
mkdir -p scripts
mkdir -p experiments/{logs,checkpoints,results}
mkdir -p notebooks
mkdir -p tests
mkdir -p docs

echo "✓ Directory structure created"

# Create __init__.py files
echo ""
echo "Creating __init__.py files..."

touch src/__init__.py
touch src/models/__init__.py
touch src/datasets/__init__.py
touch src/utils/__init__.py
touch src/training/__init__.py
touch tests/__init__.py

echo "✓ __init__.py files created"

# Create .gitkeep files to preserve empty directories
echo ""
echo "Creating .gitkeep files..."

touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/splits/.gitkeep
touch experiments/logs/.gitkeep
touch experiments/checkpoints/.gitkeep
touch experiments/results/.gitkeep

echo "✓ .gitkeep files created"

# Create virtual environment
echo ""
read -p "Create virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
    
    echo ""
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "✓ Virtual environment activated"
    
    # Install requirements
    echo ""
    read -p "Install requirements? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing requirements..."
        pip install --upgrade pip
        pip install -r requirements.txt
        echo "✓ Requirements installed"
    fi
fi

# Test installation
echo ""
read -p "Run installation tests? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running installation tests..."
    python test_installation.py
fi

# Print next steps
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Project structure created successfully!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Download dataset:"
echo "   python scripts/preprocess_data.py --dataset loveda --download_instructions"
echo "3. Place dataset in data/raw/ directory"
echo "4. Preprocess data:"
echo "   python scripts/preprocess_data.py --dataset loveda --data_dir ./data/raw"
echo "5. Update config/default.yaml with your dataset path"
echo "6. Start training:"
echo "   python scripts/train.py --config config/default.yaml"
echo ""
echo "For more information, see README.md"
echo "=========================================="