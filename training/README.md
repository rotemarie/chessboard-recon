# Training Module

Deep learning models for chess piece classification.

## 📁 Module Structure

```
training/
├── model.py          # Model architectures (ResNet18, ResNet50, VGG16)
├── train.py          # Training script with early stopping
├── evaluate.py       # Evaluation and metrics
├── utils.py          # Data loading and visualization helpers
├── __init__.py       # Package init
└── README.md         # This file
```

## 🚀 Quick Start

### 1. Train a Model

```bash
python train.py \
  --data-dir ../dataset \
  --model resnet18 \
  --epochs 100 \
  --patience 10
```

### 2. Evaluate the Model

```bash
python evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --model resnet18 \
  --data-dir ../dataset \
  --split val \
  --output-dir results
```

### 3. Extract Plots

```bash
cd ..
python extract_plots.py
```

## 📚 Module Documentation

### `model.py`

Model architectures and utilities.

**Key Functions**:
- `get_model(model_name, num_classes, pretrained, freeze_backbone)` - Create a model
- `load_model(model_path, model_name, num_classes, device)` - Load trained model
- `count_parameters(model)` - Count model parameters

**Supported Models**:
- `resnet18` - ResNet18 (11M parameters)
- `resnet50` - ResNet50 (23M parameters)
- `vgg16` - VGG16 (138M parameters)

**Example**:
```python
from training.model import get_model, load_model

# Create new model
model = get_model('resnet18', num_classes=13, pretrained=True)

# Load trained model
model = load_model('checkpoints/best_model.pth', 'resnet18')
```

### `train.py`

Training script with full CLI interface.

**Key Features**:
- Early stopping with patience
- Learning rate scheduling (StepLR)
- Automatic class balancing (weighted sampling)
- Data augmentation (random flips)
- Comet.ml experiment tracking (optional)
- Checkpoint saving (best + final)

**Usage**:
```bash
python train.py --help  # See all options

# Basic training
python train.py --data-dir ../dataset --model resnet18

# Advanced training
python train.py \
  --data-dir ../dataset \
  --model resnet50 \
  --freeze-backbone \
  --batch-size 32 \
  --epochs 50 \
  --patience 5 \
  --lr 0.01 \
  --experiment-name "resnet50_transfer"
```

**Key Arguments**:
- `--data-dir` - Root directory with train/ and val/ folders
- `--model` - Model architecture (resnet18, resnet50, vgg16)
- `--freeze-backbone` - Transfer learning mode
- `--batch-size` - Batch size (default: 16)
- `--epochs` - Maximum epochs (default: 100)
- `--patience` - Early stopping patience (default: 10)
- `--lr` - Learning rate (default: 0.001)
- `--experiment-name` - Comet.ml experiment name

### `evaluate.py`

Comprehensive evaluation with visualizations.

**Key Features**:
- Classification report (precision, recall, F1)
- Confusion matrix (raw and normalized)
- Per-class metrics visualization
- Confidence distribution analysis
- OOD detection analysis (with clean/occluded sets)

**Usage**:
```bash
# Basic evaluation
python evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --model resnet18 \
  --data-dir ../dataset \
  --split val

# With OOD analysis
python evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --model resnet18 \
  --data-dir ../dataset \
  --split val \
  --clean-dir ../data/val-no-occlusions \
  --occluded-dir ../data/val-occluded \
  --output-dir evaluation_results
```

**Output Files**:
- `{split}_classification_report.txt` - Text report with metrics
- `{split}_confusion_matrix.png` - Confusion matrix
- `{split}_confusion_matrix_normalized.png` - Normalized confusion matrix
- `{split}_per_class_metrics.png` - Per-class precision/recall/F1
- `{split}_confidence_distribution.png` - Confidence scores by class
- `ood_confidence_distribution.png` - OOD analysis (if enabled)

### `utils.py`

Data loading and visualization utilities.

**Key Functions**:
- `load_datasets(data_dir, augment_train)` - Load train/val datasets
- `get_dataloaders(image_datasets, batch_size, use_weighted_sampler)` - Create data loaders
- `create_weighted_sampler(dataset, num_samples)` - Balance class distribution
- `get_dataset_histogram(dataset)` - Get class distribution
- `plot_dataset_sample(N, dataset, save_path)` - Visualize samples
- `visualize_model_predictions(model, dataloader, ...)` - Show predictions
- `set_seed(seed)` - Set random seed for reproducibility

**Example**:
```python
from training.utils import load_datasets, get_dataloaders, set_seed

# Set seed for reproducibility
set_seed(42)

# Load datasets
image_datasets, dataset_sizes, class_names = load_datasets(
    '../dataset', augment_train=True
)

# Create dataloaders
dataloaders = get_dataloaders(
    image_datasets,
    batch_size=16,
    use_weighted_sampler=True
)
```

## 🎯 Training Modes

### Fine-Tuning (Recommended)

Train all layers from pretrained weights:

```bash
python train.py --data-dir ../dataset --model resnet18
```

**Pros**: Best accuracy  
**Cons**: Slower training, more memory  
**When to use**: Sufficient data (~30K+ samples)

### Transfer Learning

Freeze backbone, train only final layer:

```bash
python train.py \
  --data-dir ../dataset \
  --model resnet18 \
  --freeze-backbone \
  --lr 0.01
```

**Pros**: Faster training, less memory  
**Cons**: Lower accuracy potential  
**When to use**: Limited data or compute

## 📊 Expected Results

### Baseline (ResNet18, Fine-tuning)

- **Training time**: ~2-3 hours (GPU) / ~10-15 hours (CPU)
- **Validation accuracy**: ~89%
- **Per-class F1-score**: 0.85-0.95 (most classes)
- **Convergence**: ~15-20 epochs

### Class-Specific Performance

**High accuracy** (>90%):
- Empty squares
- Kings (distinct appearance)
- Queens (unique features)

**Medium accuracy** (85-90%):
- Rooks, bishops, knights
- Most pawns

**Lower accuracy** (<85%):
- Pawns in crowded positions
- Pieces with partial occlusion
- Pieces at extreme angles

## 🔧 Hyperparameter Tuning

### Learning Rate

```bash
# Lower LR for fine-tuning
python train.py --lr 0.0001  # Conservative

# Higher LR for transfer learning
python train.py --freeze-backbone --lr 0.01  # Aggressive
```

### Batch Size

```bash
# Smaller batch (more stable gradients)
python train.py --batch-size 8

# Larger batch (faster training)
python train.py --batch-size 64
```

### Augmentation

```bash
# Disable augmentation
python train.py --no-augmentation

# Disable weighted sampling
python train.py --no-weighted-sampler
```

## 📈 Experiment Tracking

### Setup Comet.ml (Optional)

```bash
# Install
pip install comet-ml

# Login
comet login  # Follow prompts

# Train with tracking
python train.py \
  --data-dir ../dataset \
  --experiment-name "my_experiment"
```

### View Results

1. Go to https://www.comet.ml
2. Navigate to your project: `chess-piece-recognition`
3. View:
   - Loss/accuracy curves
   - Hyperparameters
   - Model checkpoints
   - System metrics

## 🐛 Troubleshooting

### Out of Memory (OOM)

**Solution**: Reduce batch size
```bash
python train.py --batch-size 8
```

### Poor Accuracy (<80%)

**Solutions**:
1. Check data distribution: `python extract_plots.py`
2. Verify data splits: `ls dataset/train/ dataset/val/`
3. Try different model: `--model resnet50`
4. Increase epochs: `--epochs 200 --patience 20`

### Training Too Slow

**Solutions**:
1. Use GPU if available
2. Enable `cudnn.benchmark`: (automatic in code)
3. Increase batch size: `--batch-size 32`
4. Use smaller model: `--model resnet18`
5. Try transfer learning: `--freeze-backbone`

### Model Not Improving

**Solutions**:
1. Check learning rate: Try `--lr 0.0001` or `--lr 0.01`
2. Disable augmentation: `--no-augmentation`
3. Increase patience: `--patience 20`
4. Check data quality: Visualize samples

## 📚 Related Files

- **Main README**: [`../README.md`](../README.md) - Project overview
- **Getting Started**: [`../GETTING_STARTED.md`](../GETTING_STARTED.md) - Setup guide
- **OOD Detection**: [`../OOD_DETECTION_GUIDE.md`](../OOD_DETECTION_GUIDE.md) - Occlusion handling
- **Notebook**: [`../Chess_Piece_Recognition.ipynb`](../Chess_Piece_Recognition.ipynb) - Original code

## 🤝 Contributing

When modifying this module:
1. Follow existing code style (docstrings, type hints)
2. Test on small dataset first
3. Update this README
4. Document new hyperparameters
5. Add example usage

## 📝 Notes

### Design Decisions

1. **SGD optimizer**: More stable than Adam for fine-tuning
2. **StepLR scheduler**: Decay learning rate every 7 epochs
3. **Early stopping**: Prevents overfitting
4. **Weighted sampling**: Balances class distribution
5. **Checkpointing**: Saves best model by validation accuracy

### Implementation Details

- Uses ImageNet pretrained weights
- Normalizes with ImageNet mean/std
- Resizes images to 224×224 (standard for pretrained models)
- Applies random horizontal/vertical flips for augmentation
- Saves both best (by val accuracy) and final models

---

**For detailed usage examples, see the main [README](../README.md).**

