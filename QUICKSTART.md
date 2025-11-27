# Quick Start Guide

Get up and running with point-supervised segmentation in 5 steps!

## 🚀 Quick Setup (5 minutes)

### Step 1: Clone/Download Project

```bash
# If using git
git clone https://github.com/amrabdalbadee/remote-sensing-point-segmentation/
cd remote-sensing-point-segmentation

# Or download and extract the project files
```

### Step 2: Run Automated Setup

**On Linux/Mac:**
```bash
chmod +x setup.sh
bash setup.sh
```

**On Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python3 test_installation.py
```

You should see: `🎉 All tests passed! Your installation is ready.`

### Step 4: Download Dataset

**Get download instructions:**
```bash
python3 scripts/preprocess_data.py --dataset loveda --download_instructions
```

**Download LoveDA dataset:**
1. Visit: https://github.com/Junjue-Wang/LoveDA
2. Download from Google Drive or Baidu Netdisk
3. Extract to `./data/raw/LoveDA/`

**Expected structure:**
```
data/raw/LoveDA/
├── Train/
│   ├── Urban/
│   │   ├── images_png/
│   │   └── masks_png/
│   └── Rural/
│       ├── images_png/
│       └── masks_png/
└── Val/
    ├── Urban/
    │   ├── images_png/
    │   └── masks_png/
    └── Rural/
        ├── images_png/
        └── masks_png/
```

### Step 5: Preprocess Data

```bash
python3 scripts/preprocess_data.py --dataset loveda --data_dir ./data/raw/LoveDA
```

This will:
- Verify dataset structure
- Compute statistics
- Generate sample visualizations

---

## 🎯 Training Your First Model (10 minutes)

### Quick Training Run

```bash
# Update config with your data path
# Edit config/default.yaml: data.root_dir = "./data/raw/LoveDA"

# Start training (with 5 points per class, random sampling)
python3 scripts/train.py --config config/default.yaml
```

**What to expect:**
- Training will start immediately
- Progress bars show epoch progress
- Checkpoints saved in `experiments/baseline/checkpoints/`
- Best model saved as `best_model.pth`

**Training output:**
```
Epoch 1/100
  Train Loss: 1.2345
  Val Loss:   1.1234
  Val mIoU:   0.4567
  Val Dice:   0.5678
  ✓ New best model saved!
```

### Monitor Training

**Option 1: Watch console output**
- Real-time metrics displayed every 10 batches
- Validation metrics after each epoch

**Option 2: TensorBoard (optional)**
```bash
tensorboard --logdir experiments/baseline/logs
```

---

## 🔬 Running Experiments (30 minutes)

### Experiment 1: Effect of Number of Points

**Manual approach:**
```bash
# Test with 1 point per class
python3 scripts/train.py --config config/experiment_configs/exp1_num_points.yaml

# Edit config: change num_points_per_class to 5, 10, 20, etc.
# Repeat training
```

**Automated approach:**
```bash
# Runs all configurations automatically
python3 scripts/run_experiments.py --experiment num_points
```

This will:
- Train models with [1, 3, 5, 10, 20, 50] points per class
- Train full supervision baseline
- Generate comparison plots automatically
- Save results to `experiments/automated/analysis/`

### Experiment 2: Compare Sampling Strategies

```bash
# Automated comparison of random, centroid, boundary, and grid sampling
python3 scripts/run_experiments.py --experiment sampling_strategy
```

### Run All Experiments

```bash
# Run both experiments sequentially
python3 scripts/run_experiments.py --experiment all
```

**Note:** Running all experiments takes several hours (depends on GPU/dataset size)

---

## 📊 Evaluating Results

### Evaluate a Trained Model

```bash
python3 scripts/evaluate.py \
    --checkpoint experiments/baseline/checkpoints/best_model.pth \
    --config config/default.yaml \
    --output_dir ./evaluation_results
```

**Output includes:**
- `evaluation_metrics.json` - Quantitative results
- `visualizations/` - Predicted vs ground truth comparisons
- `confusion_matrix.png` - Confusion matrix
- `class_metrics.png` - Per-class IoU, Precision, Recall

### View Results

**Metrics:**
```bash
cat evaluation_results/evaluation_metrics.json
```

**Visualizations:**
```bash
# Open in your image viewer
open evaluation_results/visualizations/predictions_batch_0.png
```

---

## 📝 Writing the Technical Report

### Gather Results

After running experiments:

1. **Training curves:** `experiments/*/visualizations/training_curves.png`
2. **Comparison plots:** `experiments/automated/analysis/*.png`
3. **Evaluation metrics:** `evaluation_results/evaluation_metrics.json`
4. **Visualizations:** `evaluation_results/visualizations/`

### Report Structure

```
1. Introduction (0.5 pages)
   - Problem statement
   - Motivation for point supervision
   
2. Method (1.5 pages)
   - Partial cross-entropy loss (include formula)
   - Point sampling strategies (describe each)
   - Network architecture (U-Net diagram)
   - Implementation details
   
3. Experiments (2-3 pages)
   
   Experiment 1: Number of Points
   - Purpose: Investigate annotation efficiency
   - Hypothesis: More points → better performance (diminishing returns)
   - Setup: Tested [1, 3, 5, 10, 20, 50] points, compared to full supervision
   - Results: Table + plot from experiments/automated/analysis/
   - Analysis: Discuss optimal trade-off
   
   Experiment 2: Sampling Strategies
   - Purpose: Compare placement strategies
   - Hypothesis: Strategic placement > random
   - Setup: Random, Centroid, Boundary, Grid with 10 points each
   - Results: Bar chart comparing mIoU
   - Analysis: Explain why certain strategies work better
   
4. Conclusion (0.5 pages)
   - Key findings
   - Practical implications
   - Limitations and future work
```

### Include These Figures

1. **Method:**
   - Point sampling visualization (create using notebooks)
   - U-Net architecture diagram
   
2. **Results:**
   - Training curves (loss, mIoU over epochs)
   - Exp 1: Line plot (points vs mIoU)
   - Exp 2: Bar chart (strategy comparison)
   - Qualitative results (image, GT, prediction side-by-side)
   - Confusion matrix
   - Per-class metrics

### Key Tables

**Table 1: Experiment 1 Results**
| Points per Class | mIoU | Dice | Pixel Acc | Training Time |
|------------------|------|------|-----------|---------------|
| 1 | ... | ... | ... | ... |
| 5 | ... | ... | ... | ... |
| Full Supervision | ... | ... | ... | ... |

**Table 2: Experiment 2 Results**
| Strategy | mIoU | Dice | Points Sampled |
|----------|------|------|----------------|
| Random | ... | ... | ... |
| Centroid | ... | ... | ... |
| Boundary | ... | ... | ... |
| Grid | ... | ... | ... |

---

## 💡 Tips for Excellence

### Code Quality
- ✅ All code is already modular and documented
- ✅ Follow PEP 8 style (use `black` formatter if needed)
- ✅ Include docstrings (already provided)

### Experiments
- 🔥 **Run multiple seeds** (3-5) and report mean ± std
- 🔥 **Statistical significance**: Use t-tests when comparing methods
- 🔥 **Ablation study**: Show each component matters

### Report Quality
- 📊 High-quality figures (300 DPI, clear labels)
- 📝 Clear writing (no jargon without explanation)
- 🔢 Report all important hyperparameters
- 🎯 Answer: "Could someone reproduce this?"

### Going Above and Beyond
1. **Add confidence intervals** to plots
2. **Analyze failure cases** - show where the model struggles
3. **Computational efficiency** - report training time, memory usage
4. **Additional experiments:**
   - Class imbalance handling
   - Different network architectures
   - Data augmentation impact

---

## 🐛 Troubleshooting

### Common Issues

**"CUDA out of memory"**
```yaml
# In config/default.yaml, reduce:
training:
  batch_size: 4  # Was 8
data:
  img_size: 128  # Was 256
```

**"Dataset not found"**
```bash
# Verify path in config/default.yaml
data:
  root_dir: "./data/raw/LoveDA"  # Check this path

# Check dataset structure
ls -R data/raw/LoveDA/
```

**"Module not found"**
```bash
# Reinstall packages
pip install -r requirements.txt

# Ensure __init__.py files exist
find src -type d -exec touch {}/__init__.py \;
```

**Slow training**
```yaml
# Increase workers (if you have CPU cores available)
training:
  num_workers: 8  # Was 4

# Use lighter model
model:
  type: "light_unet"  # Was "unet"
```

---

## 📚 Additional Resources

- **LoveDA Paper:** https://arxiv.org/abs/2110.08733
- **U-Net Paper:** https://arxiv.org/abs/1505.04597
- **Point Supervision Survey:** Search for weakly-supervised segmentation papers

---

## ✅ Checklist Before Submission

- [ ] Code runs without errors
- [ ] All experiments completed
- [ ] Results saved and organized
- [ ] Technical report written
- [ ] Figures and tables included
- [ ] Code is well-documented
- [ ] Requirements.txt included
- [ ] README.md updated
- [ ] Deliverables packaged:
  - Python files or Jupyter notebook ✓
  - Technical report (PDF) ✓
  - Supporting documents ✓

---

**You're all set! Good luck with your technical assessment! 🚀**