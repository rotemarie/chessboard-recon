# Getting Started with Chessboard Preprocessing

## ⚡ Quick Start (3 Commands)

```bash
# 1. Setup environment
python3 -m venv bguenv && source bguenv/bin/activate
pip install --index-url https://pypi.org/simple -r requirements.txt

# 2. Test pipeline
cd preprocessing && python test_pipeline.py

# 3. Run full preprocessing
python preprocess_data.py
```

---

## 📋 What This Does

This preprocessing pipeline converts raw chessboard images into a labeled dataset ready for training:

**Input:** Raw chessboard images (angled, varying lighting) + CSV files with FEN labels

**Processing:**
1. **Board Detection** → Find corners, apply perspective transform to get 512×512 top-down view
2. **Square Extraction** → Slice into 64 squares (64×64 pixels each)
3. **FEN Parsing** → Label each square with piece type (13 classes)
4. **Organization** → Save squares organized by class

**Output:** ~33,000 labeled square images ready for training

---

## 🔧 Detailed Setup

### Prerequisites
- Python 3.8+
- macOS/Linux/Windows
- ~10 GB free disk space

### Step 0: Download Data (First Time Only)

⚠️ **Important:** The raw data is NOT in this repository due to its size.

**Download the dataset:**
1. See [`data/README.md`](data/README.md) for download link
2. Extract to the `data/` folder
3. Verify: `ls data/per_frame/` should show game folders

**Quick check:**
```bash
# Should see 5 game folders
ls data/per_frame/
# Output: game2_per_frame  game4_per_frame  game5_per_frame  game6_per_frame  game7_per_frame
```

👉 **Don't have the data?** Ask your team for the shared drive link or see `data/README.md`

### Step 1: Create Virtual Environment

```bash
cd "/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon"

# Create virtual environment
python3 -m venv bguenv

# Activate it
source bguenv/bin/activate  # macOS/Linux
# bguenv\Scripts\activate   # Windows
```

### Step 2: Install Dependencies

```bash
# Use PyPI directly (bypasses corporate repos)
pip install --index-url https://pypi.org/simple -r requirements.txt
```

**Troubleshooting:** If you get SSL or timeout errors, the `--index-url` flag forces pip to use the standard PyPI repository.

### Step 3: Test the Pipeline

```bash
cd preprocessing
python test_pipeline.py
```

**Expected output:**
```
============================================================
TEST SUMMARY
============================================================
✓ PASS: Board Detection
✓ PASS: Square Extraction
✓ PASS: FEN Parsing
✓ PASS: Full Pipeline

4/4 tests passed

🎉 All tests passed! Ready to run full preprocessing.
```

### Step 4: Run Full Preprocessing

```bash
python preprocess_data.py
```

**What happens:**
- Processes 5 games (~517 labeled frames)
- Extracts ~33,000 square images
- Takes 5-10 minutes
- Shows progress bars and statistics

**Expected output:**
```
============================================================
Processing game2
============================================================
Found 77 labeled frames
game2: 100%|████████████████| 77/77 [00:45<00:00]

game2 Statistics:
  Success: 75
  Failed: 2
  Total squares extracted: 4,800

============================================================
FINAL STATISTICS
============================================================
Total frames processed successfully: 464
Total square images extracted: 29,696

============================================================
CLASS DISTRIBUTION
============================================================
  empty               : 14,832 images
  white_pawn         :  3,712 images
  black_pawn         :  3,648 images
  ...
```

---

## 📊 What Gets Created

```
preprocessed_data/
├── train/                      # 📁 Labeled squares by class
│   ├── white_pawn/            # ~3,700 images
│   ├── black_knight/          # ~1,400 images
│   ├── empty/                 # ~14,800 images
│   └── ... (13 classes total)
│
├── warped_boards/             # 🔍 Top-down board views for inspection
│   ├── game2_frame_000200.jpg
│   └── ...
│
├── failed_detections/         # ⚠️ Frames where detection failed
│   └── ...
│
└── metadata/                  # 📝 Processing logs (CSV)
    ├── game2_metadata.csv
    └── ...
```

---

## ✅ Verify Results

### 1. Visual Inspection

```bash
# Check warped boards (should be square, flat, no distortion)
open preprocessed_data/warped_boards/game2_frame_000200.jpg

# Check some squares (should be 64×64, pieces centered)
open preprocessed_data/train/white_pawn/game2_frame_000200_a2.jpg
open preprocessed_data/train/black_knight/game2_frame_000200_b8.jpg
open preprocessed_data/train/empty/game2_frame_000200_e4.jpg
```

### 2. Check Class Distribution

```bash
cd preprocessed_data/train
for dir in */; do
    count=$(ls -1 "$dir" | wc -l)
    printf "%-20s: %6d images\n" "${dir%/}" "$count"
done
```

### 3. Review Failed Detections

```bash
ls -lh preprocessed_data/failed_detections/
cat preprocessed_data/metadata/game2_metadata.csv
```

---

## 🐛 Troubleshooting

### ModuleNotFoundError: No module named 'cv2'
**Solution:** Make sure virtual environment is activated and dependencies are installed
```bash
source bguenv/bin/activate
pip install --index-url https://pypi.org/simple -r requirements.txt
```

### Many frames in failed_detections/
**Solution:** Adjust detection thresholds in `preprocessing/board_detector.py`
- In `_find_board_corners()`, modify Canny thresholds:
  ```python
  edges = cv2.Canny(blurred, 50, 150)  # Original
  edges = cv2.Canny(blurred, 30, 100)  # Try this for darker images
  ```

### Squares look misaligned
**Solution:** Run with debug mode to visualize corner detection
```python
from preprocessing.board_detector import BoardDetector
detector = BoardDetector(board_size=512)
warped = detector.detect_board(image, debug=True)  # Shows corners and edges
```

### pip timeout errors
**Solution:** Use `--index-url` flag to bypass custom repositories
```bash
pip install --index-url https://pypi.org/simple -r requirements.txt
```

---

## 🎨 Visualize the Pipeline (Optional)

Want to see how it works step-by-step? Run this in Python:

```python
import cv2
from preprocessing.board_detector import BoardDetector
from preprocessing.square_extractor import SquareExtractor, FENParser

# Load sample image
image = cv2.imread("data/per_frame/game2_per_frame/tagged_images/frame_000200.jpg")

# Step 1: Detect and warp board
detector = BoardDetector(board_size=512)
warped = detector.detect_board(image, debug=True)  # Shows visualization!

# Step 2: Extract squares
extractor = SquareExtractor(board_size=512)
squares = extractor.extract_squares(warped)

# Step 3: Parse FEN
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
labels = FENParser.fen_to_labels(fen)

# Step 4: Visualize results
extractor.visualize_squares(squares, labels, num_display=16)
```

---

## 🎯 Next Steps - Detailed Roadmap

### Phase 0: Complete Preprocessing ✅

**What:** Process raw images into labeled squares

**Status:** ✅ DONE

**Output:** `preprocessed_data/train/` with ~33,000 labeled squares

---

### Phase 1: Dataset Preparation (15-30 minutes)

#### Step 1.1: Split Dataset by Game 🔴 DO THIS NEXT

**Why:** CRITICAL to prevent data leakage! Frames from the same game are highly correlated.

**How:**
```bash
cd preprocessing
python split_dataset.py
```

**What it does:**
- Splits by GAME, not by frame (prevents leakage)
- Creates `dataset/train/`, `dataset/val/`, `dataset/test/`
- Saves split info for reproducibility
- Typical split: 70% train, 15% val, 15% test

**Output:** `dataset/` with train/val/test folders

**Verify:**
```bash
ls dataset/
# Should see: train/ val/ test/ split_info.json

# Check split info
cat dataset/split_info.json
```

---

### Phase 2: Research & Design (1-2 days)

#### Step 2.1: Research OOD Detection Methods

**Why:** Occlusion handling is a CORE requirement, not optional!

**What to research:**
1. **Baseline: Confidence Thresholding**
   - Use softmax probability < threshold → "unknown"
   - Pros: Simple, no extra training
   - Cons: Poorly calibrated, doesn't detect true OOD

2. **ODIN (Out-of-Distribution detector for Neural networks)**
   - Paper: https://arxiv.org/abs/1706.02690
   - Uses temperature scaling + input preprocessing
   - Pros: Effective, minimal overhead
   - Cons: Requires hyperparameter tuning

3. **Mahalanobis Distance**
   - Paper: https://arxiv.org/abs/1807.03888
   - Measures distance in feature space
   - Pros: Theoretically sound, works well
   - Cons: Requires storing statistics per class

4. **OpenMax**
   - Paper: https://arxiv.org/abs/1511.06233
   - Replaces softmax with open-set aware layer
   - Pros: Principled approach
   - Cons: More complex implementation

5. **Energy-Based OOD**
   - Paper: https://arxiv.org/abs/2010.03759
   - Uses energy scores instead of softmax
   - Pros: Simple, effective
   - Cons: Requires understanding energy functions

**Recommendation:** Start with ODIN or Mahalanobis (good balance of effectiveness and complexity)

**👉 See [OOD_DETECTION_GUIDE.md](OOD_DETECTION_GUIDE.md) for detailed explanations, code examples, and recommendations**

**Output:** Decision on which method(s) to implement

#### Step 2.2: Design Model Architecture

**Considerations:**
1. **Input:** 64×64 RGB images
2. **Output:** 13 classes (or 14 with "unknown"?)
3. **Backbone:** 
   - ResNet18/34 (good balance)
   - EfficientNet-B0 (efficient)
   - MobileNetV3 (fast inference)
   - Custom CNN (if needed)

4. **OOD Integration:**
   - Extract features before final layer (for Mahalanobis)
   - Add temperature parameter (for ODIN)
   - Ensure proper calibration

**Questions to answer:**
- [ ] Use pretrained weights (ImageNet)? → Recommended YES
- [ ] Fine-tune entire network or just classifier? → Start with just classifier
- [ ] Add dropout for calibration? → YES
- [ ] Use label smoothing? → YES, helps calibration

**Output:** Architecture diagram/code skeleton

#### Step 2.3: Plan Data Augmentation Strategy

**Required augmentations:**
1. **Geometric:** Rotation (±10°), slight scaling
2. **Color:** Brightness, contrast, saturation adjustments
3. **Simulated occlusions:** 
   - Add random patches/rectangles
   - Blur regions
   - This is KEY for OOD training!

**Libraries:**
- Albumentations (recommended, fast)
- torchvision.transforms
- Custom occlusion augmentations

**Output:** Augmentation pipeline code

---

### Phase 3: Implementation & Training (3-5 days)

#### Step 3.1: Create Data Loaders

**Files to create:**
```
training/
├── dataset.py          # ChessSquareDataset class
├── augmentations.py    # Augmentation pipelines
└── dataloader.py       # Create train/val/test loaders
```

**Implementation:**
```python
class ChessSquareDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        # Load images from class folders
        # Apply transforms
        pass
```

**Include:**
- Class weights for imbalanced classes
- Proper transforms for train vs val/test
- Efficient loading (consider caching)

**Test:** Visualize batches to verify augmentations

#### Step 3.2: Implement Model

**Files to create:**
```
training/
├── model.py            # Model architecture
├── ood_detector.py     # OOD detection logic
└── config.py           # Hyperparameters
```

**Model features:**
- Feature extraction hook (for OOD)
- Temperature parameter (if using ODIN)
- Proper initialization
- Model summary/parameter count

#### Step 3.3: Implement Training Loop

**Files to create:**
```
training/
├── train.py            # Main training script
├── losses.py           # Loss functions (maybe custom)
└── metrics.py          # Evaluation metrics
```

**Training components:**
1. **Loss function:**
   - CrossEntropyLoss with label smoothing
   - Class weights for imbalance
   
2. **Optimizer:**
   - Adam or AdamW (start with lr=1e-3)
   - Cosine annealing schedule
   
3. **Metrics:**
   - Per-class accuracy
   - Confusion matrix
   - Top-1 and Top-2 accuracy
   
4. **Logging:**
   - TensorBoard or Weights & Biases
   - Save checkpoints
   - Early stopping

**Run training:**
```bash
python training/train.py --config configs/baseline.yaml
```

**Expected results:**
- Train accuracy: 95-98%
- Val accuracy: 85-92% (depending on game similarity)
- Some classes (pawns, empty) should be near-perfect
- Some classes (bishops, knights) might be confused

#### Step 3.4: Implement & Tune OOD Detector

**Files to create:**
```
training/
└── ood_detector.py     # OOD detection implementation
```

**Implementation steps:**
1. **If using ODIN:**
   - Add temperature parameter
   - Implement input preprocessing
   - Tune temperature and epsilon on val set
   
2. **If using Mahalanobis:**
   - Extract features for all training samples
   - Compute class-wise mean and covariance
   - Implement distance calculation
   - Tune threshold on val set

**Create test set with occlusions:**
- Manually occlude some val images
- Test OOD detector performance
- Tune threshold: balance false positives vs false negatives

**Metrics:**
- AUROC (Area Under ROC Curve)
- FPR at 95% TPR
- Detection accuracy

---

### Phase 4: Board Reconstruction (2-3 days)

#### Step 4.1: Implement Full Pipeline

**Files to create:**
```
inference/
├── pipeline.py         # Full inference pipeline
├── board_predictor.py  # Combines all components
└── visualize.py        # Visualization utilities
```

**Pipeline flow:**
```
Input Image
    ↓
[Board Detector]  ← From preprocessing
    ↓
Warped Board (512×512)
    ↓
[Square Extractor]  ← From preprocessing
    ↓
64 Square Images (64×64 each)
    ↓
[Model + OOD Detector]  ← Trained model
    ↓
64 Predictions + Confidence/OOD Scores
    ↓
[FEN Generator]  ← Convert to FEN
    ↓
FEN String (with '?' for unknowns)
```

**Implementation:**
```python
class BoardPredictor:
    def __init__(self, model_path, ood_detector):
        self.board_detector = BoardDetector()
        self.square_extractor = SquareExtractor()
        self.model = load_model(model_path)
        self.ood_detector = ood_detector
    
    def predict(self, image):
        # 1. Detect board
        warped = self.board_detector.detect_board(image)
        
        # 2. Extract squares
        squares = self.square_extractor.extract_squares(warped)
        
        # 3. Classify with OOD detection
        predictions = []
        for square in squares:
            pred, is_ood = self.classify_with_ood(square)
            predictions.append('?' if is_ood else pred)
        
        # 4. Generate FEN
        fen = FENParser.labels_to_fen(predictions)
        
        return fen, predictions
```

#### Step 4.2: Visualize Predictions

**Create visualization tool:**
```python
def visualize_prediction(image, predictions, ground_truth=None):
    # Show:
    # - Original image
    # - Warped board with predictions overlay
    # - FEN string
    # - Differences from ground truth (if available)
    # - Confidence scores
    # - OOD detections highlighted
    pass
```

**Usage:**
```bash
python inference/visualize.py --image data/test_image.jpg --output results/
```

---

### Phase 5: Evaluation (1-2 days)

#### Step 5.1: Per-Square Evaluation

**Metrics to compute:**
1. **Accuracy:** Overall and per-class
2. **Precision/Recall/F1:** Per class
3. **Confusion Matrix:** Which pieces are confused?
4. **OOD Detection:**
   - True positive rate on occluded squares
   - False positive rate on normal squares

**Files to create:**
```
evaluation/
├── evaluate.py         # Evaluation script
├── metrics.py          # Metric calculations
└── analyze.py          # Error analysis
```

**Run evaluation:**
```bash
python evaluation/evaluate.py --test-dir dataset/test/ --model-path checkpoints/best.pth
```

#### Step 5.2: Board-Level Evaluation

**Metrics:**
1. **Exact Match:** Full board FEN exactly correct
2. **Square Accuracy:** What % of 64 squares correct?
3. **Piece Accuracy:** Excluding empty squares
4. **Critical Errors:** King/Queen misclassifications

**Expected results:**
- Square accuracy: 85-95%
- Board exact match: 40-70% (strict!)
- Empty square accuracy: >95%
- Piece accuracy: 80-90%

#### Step 5.3: Error Analysis

**Questions to answer:**
1. Which pieces are most confused? (e.g., bishop ↔ knight?)
2. Which games perform worst? (lighting? angle?)
3. Are errors clustered on board edges?
4. How many false OOD detections?
5. How many missed occlusions?

**Create error report:**
- Most confused pairs
- Hardest samples
- Failure modes
- Recommendations for improvement

---

### Phase 6: Improvements & Iteration (Ongoing)

#### Possible improvements:

1. **Data:**
   - Add more games from PGN folder
   - Use temporal information for semi-supervised learning
   - Generate synthetic data

2. **Model:**
   - Try different architectures
   - Ensemble multiple models
   - Use test-time augmentation

3. **OOD:**
   - Combine multiple OOD methods
   - Train auxiliary OOD classifier
   - Use uncertainty quantification

4. **Post-processing:**
   - Use chess rules (only 1 king per side)
   - Check valid positions
   - Beam search for most likely board states

---

## 📋 Summary Checklist

### Immediate (Now)
- [ ] Run preprocessing (if not done): `python preprocess_data.py`
- [ ] **Split dataset: `python split_dataset.py`** ← START HERE

### Phase 1: Research (1-2 days)
- [ ] Read ODIN and Mahalanobis papers
- [ ] Decide on OOD method
- [ ] Choose model architecture
- [ ] Design augmentation strategy

### Phase 2: Implementation (3-5 days)
- [ ] Implement dataset and dataloaders
- [ ] Implement model architecture
- [ ] Implement training loop
- [ ] Train baseline model
- [ ] Implement OOD detector
- [ ] Tune OOD threshold

### Phase 3: Integration (2-3 days)
- [ ] Build full inference pipeline
- [ ] Create visualization tools
- [ ] Test on sample images

### Phase 4: Evaluation (1-2 days)
- [ ] Evaluate on test set
- [ ] Compute all metrics
- [ ] Perform error analysis
- [ ] Write results report

### Phase 5: Iteration (Ongoing)
- [ ] Identify failure modes
- [ ] Implement improvements
- [ ] Re-evaluate

**Total estimated time:** 2-3 weeks for complete implementation

---

## 📚 Additional Documentation

- **Project Overview**: [README.md](README.md) - Complete project documentation
- **OOD Detection**: [OOD_DETECTION_GUIDE.md](OOD_DETECTION_GUIDE.md) - ⭐ Guide to handling occlusions
- **Module Details**: [preprocessing/README.md](preprocessing/README.md) - API reference
- **Pipeline Diagram**: [preprocessing/pipeline_diagram.txt](preprocessing/pipeline_diagram.txt) - Visual explanation
- **Implementation**: [preprocessing/IMPLEMENTATION_SUMMARY.md](preprocessing/IMPLEMENTATION_SUMMARY.md) - What was built

---

## 💡 What Was Created

This preprocessing pipeline includes:

**Core Modules:**
- `board_detector.py` - Board detection with edge detection + fallback method
- `square_extractor.py` - Square extraction and FEN parsing
- `preprocess_data.py` - Main pipeline with error handling

**Testing:**
- `test_pipeline.py` - 4 comprehensive tests

**Documentation:**
- Multiple README files with detailed explanations
- Visual pipeline diagram
- Troubleshooting guides

**Configuration:**
- `requirements.txt` - All dependencies
- `.gitignore` - Git ignore rules

All code is documented with comments explaining tricky parts!

