# Chessboard Recognition: Square Classification and Board-State Reconstruction

Deep learning project for classifying chess pieces in board squares and reconstructing complete board states from images.

## 📋 Project Overview

Given real chessboard images, this system:
1. **Detects** the chessboard and extracts a top-down view
2. **Classifies** each of the 64 squares into piece classes (white pawn, black knight, empty, etc.)
3. **Reconstructs** the chess board state in FEN notation
4. **Handles occlusions** by detecting and marking unknown squares

## 🎯 Key Features

- **Robust Board Detection**: Works with angled photos and varying lighting
- **Accurate Square Classification**: 13-class classifier (12 pieces + empty)
- **Occlusion Handling**: Detects occluded/uncertain squares
- **FEN Output**: Generates standard chess notation for board state
- **Temporal Independence**: Classifies from single static images (no video context)

## 📁 Project Structure

```
chessboard-recon/
├── data/                           # Raw data
│   ├── per_frame/                  # Labeled games (CSV + images)
│   │   ├── game2_per_frame/
│   │   ├── game4_per_frame/
│   │   ├── game5_per_frame/
│   │   ├── game6_per_frame/
│   │   └── game7_per_frame/
│   ├── PGN/                        # Games with PGN files
│   │   ├── c06/ (game8, game9, game10)
│   │   └── c17/ (game11, game12, game13)
│   └── README.md                   # Data download instructions
│
├── preprocessing/                  # Data preprocessing pipeline
│   ├── board_detector.py          # Board detection & warping
│   ├── square_extractor.py        # Square extraction & FEN parsing
│   ├── preprocess_data.py         # Main preprocessing script
│   ├── split_dataset.py           # Train/val/test splitting
│   ├── create_padded_dataset.py   # Create padded squares dataset
│   ├── test_pipeline.py           # Testing script
│   ├── pipeline_diagram.txt       # Visual pipeline explanation
│   └── README.md                  # Preprocessing documentation
│
├── preprocessed_data/             # Processed data (created after running)
│   └── train/                     # All labeled squares (before split)
│       ├── white_pawn/
│       ├── black_knight/
│       ├── empty/
│       └── ... (13 classes total)
│
├── dataset/                       # Split dataset (after split_dataset.py)
│   ├── train/                     # 70% of data
│   ├── val/                       # 15% of data
│   └── test/                      # 15% of data
│
├── training/                      # Model training and evaluation
│   ├── model.py                   # Model architectures
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation and metrics
│   ├── utils.py                   # Data loading and helpers
│   └── __init__.py                # Package init
│
├── plots/                         # Generated plots (from extract_plots.py)
│   ├── 01_train_val_distribution.png
│   ├── 02_train_val_histogram.png
│   └── ... (evaluation plots)
│
├── checkpoints/                   # Model checkpoints (created during training)
│   ├── best_model.pth            # Best model by validation accuracy
│   └── final_model.pth           # Final model after training
│
├── Chess_Piece_Recognition.ipynb # Original Jupyter notebook
├── extract_plots.py              # Extract plots from notebook results
├── requirements.txt              # Python dependencies
├── GETTING_STARTED.md            # Detailed setup and roadmap
├── OOD_DETECTION_GUIDE.md        # Out-of-distribution detection guide
└── README.md                     # This file
```

## 🚀 Quick Start

```bash
# 0. Download data (see data/README.md for link)
# Extract to data/ folder

# 1. Setup environment
python3 -m venv bguenv && source bguenv/bin/activate
pip install --index-url https://pypi.org/simple -r requirements.txt

# 2. Test pipeline
cd preprocessing && python test_pipeline.py

# 3. Run preprocessing
python preprocess_data.py

# 4. Split dataset
python split_dataset.py

# 5. Train model
cd ../training
python train.py --data-dir ../dataset --model resnet18 --epochs 100

# 6. Evaluate model
python evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --model resnet18 \
  --data-dir ../dataset \
  --split val

# 7. Extract plots for reports
cd .. && python extract_plots.py
```

⚠️ **Data not included:** Raw images (~3-6 GB) are hosted externally. See [`data/README.md`](data/README.md) for download instructions.


## 🔧 Preprocessing Pipeline

### Step 1: Board Localization & Warping

**Purpose**: Find the chessboard and apply perspective transform

**Method**:
1. Convert to grayscale and apply Gaussian blur
2. Canny edge detection
3. Find contours and filter for quadrilaterals
4. Order corners (top-left, top-right, bottom-right, bottom-left)
5. Apply perspective transform to get 512×512 top-down view

**Fallback**: Adaptive thresholding + morphological operations

**Code**: `preprocessing/board_detector.py`

```python
from preprocessing.board_detector import BoardDetector

detector = BoardDetector(board_size=512)
warped_board = detector.detect_board(image, debug=False)
```

### Step 2: Square Extraction

**Purpose**: Slice the warped board into 64 individual squares

**Method**:
1. Divide 512×512 board into 8×8 grid
2. Extract 64×64 pixel squares
3. Order: a8, b8, ..., h8, a7, ..., h1 (matches FEN order)

**Code**: `preprocessing/square_extractor.py`

```python
from preprocessing.square_extractor import SquareExtractor

extractor = SquareExtractor(board_size=512)
squares = extractor.extract_squares(warped_board)  # Returns 64 images
```

### Step 3: FEN Parsing & Labeling

**Purpose**: Convert FEN notation to per-square labels

**FEN Format**: Describes board state from rank 8 to rank 1
- Uppercase = white pieces (R, N, B, Q, K, P)
- Lowercase = black pieces (r, n, b, q, k, p)
- Numbers = consecutive empty squares
- `/` = rank separator

**Example**: `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR` (starting position)

**Code**: `preprocessing/square_extractor.py`

```python
from preprocessing.square_extractor import FENParser

labels = FENParser.fen_to_labels(fen)  # Returns 64 labels
reconstructed_fen = FENParser.labels_to_fen(labels)
```

### Step 4: Save Organized Dataset

**Purpose**: Create a structured dataset for training

**Output**:
- Each square saved to its class folder
- Filename: `game2_frame_000200_a8.jpg` (traceable to source)
- Metadata CSV with processing status
- Warped boards for visual inspection

## 🤖 Model Training

### Training Pipeline

The `training/` module provides a complete pipeline for training chess piece classifiers:

**Structure**:
```
training/
├── model.py          # Model architectures (ResNet18, ResNet50, VGG16)
├── train.py          # Training script with early stopping
├── evaluate.py       # Evaluation and metrics
├── utils.py          # Data loading and visualization
└── __init__.py       # Package init
```

### Quick Training

```bash
# 1. Split dataset (if not done already)
cd preprocessing
python split_dataset.py

# 2. Train ResNet18
cd ../training
python train.py \
  --data-dir ../dataset \
  --model resnet18 \
  --epochs 100 \
  --patience 10 \
  --experiment-name "resnet18_baseline"
```

### Training Options

**Model Architectures**:
- `resnet18` - Fast, good baseline (11M params)
- `resnet50` - Better accuracy, slower (23M params)
- `vgg16` - Alternative architecture (138M params)

**Training Modes**:
- **Fine-tuning** (default): Train all layers
  ```bash
  python train.py --data-dir ../dataset --model resnet18
  ```
- **Transfer Learning**: Freeze backbone, train only final layer
  ```bash
  python train.py --data-dir ../dataset --model resnet18 --freeze-backbone
  ```

**Key Arguments**:
```bash
python train.py \
  --data-dir ../dataset \              # Dataset directory
  --model resnet18 \                    # Architecture
  --batch-size 16 \                     # Batch size
  --epochs 100 \                        # Max epochs
  --patience 10 \                       # Early stopping patience
  --lr 0.001 \                          # Learning rate
  --no-weighted-sampler \               # Disable class balancing
  --no-augmentation \                   # Disable data augmentation
  --experiment-name "my_experiment" \   # Comet.ml name
  --checkpoint-dir ./checkpoints        # Save location
```

### Features

✅ **Automatic Class Balancing**: Weighted sampling ensures equal representation  
✅ **Data Augmentation**: Random flips for better generalization  
✅ **Early Stopping**: Prevents overfitting  
✅ **Learning Rate Scheduling**: StepLR with decay  
✅ **Experiment Tracking**: Comet.ml integration (optional)  
✅ **Checkpointing**: Saves best and final models  

### Evaluation

```bash
# Evaluate on validation set
python evaluate.py \
  --checkpoint ./checkpoints/best_model.pth \
  --model resnet18 \
  --data-dir ../dataset \
  --split val \
  --output-dir ./results

# This generates:
# - Classification report (precision, recall, F1)
# - Confusion matrix (normalized and raw)
# - Per-class metrics plot
# - Confidence distribution plot
```

**OOD Detection Analysis** (for occluded images):
```bash
python evaluate.py \
  --checkpoint ./checkpoints/best_model.pth \
  --model resnet18 \
  --data-dir ../dataset \
  --split val \
  --clean-dir ../data/val-no-occlusions \
  --occluded-dir ../data/val-occluded \
  --output-dir ./results
```

### Experiment Tracking with Comet.ml

The training script supports [Comet.ml](https://www.comet.ml/) for experiment tracking:

1. **Setup** (optional):
   ```bash
   pip install comet-ml
   comet login  # Follow prompts
   ```

2. **Train with tracking**:
   ```bash
   python train.py --data-dir ../dataset --experiment-name "my_exp"
   ```

3. **View results**: Check your Comet.ml dashboard for:
   - Training/validation loss and accuracy curves
   - Hyperparameters
   - Model checkpoints
   - System metrics (GPU, CPU, memory)

If Comet.ml is not installed, training proceeds without tracking.

### Visualization

**Extract plots from notebook**:
```bash
python extract_plots.py
```

This generates:
- `plots/01_train_val_distribution.png` - Class distribution
- `plots/02_train_val_histogram.png` - Class counts
- `plots/05_random_samples_per_class.png` - Sample images

### Training from Notebook

The original training code is in `Chess_Piece_Recognition.ipynb`. To run:

1. **Google Colab** (recommended for GPU):
   - Upload notebook to Colab
   - Upload dataset zip to Google Drive
   - Run cells sequentially

2. **Local Jupyter**:
   ```bash
   jupyter notebook Chess_Piece_Recognition.ipynb
   ```

Note: The notebook uses Google Drive paths. Modify paths for local execution.

## 📊 Data Statistics

### Labeled Games (per_frame)
- **game2**: 77 labeled frames
- **game4**: 184 labeled frames
- **game5**: 109 labeled frames
- **game6**: 92 labeled frames
- **game7**: 55 labeled frames
- **Total**: ~517 labeled frames → ~33,088 labeled squares

### Unlabeled Games (PGN)
- **game8-10**: ~27,534 frames
- **game11-13**: ~38,771 frames
- **Total**: ~66,305 frames (can be labeled using PGN + temporal tracking)

### Piece Classes (13 total)
1. `empty` - Empty square
2. `white_pawn`
3. `white_knight`
4. `white_bishop`
5. `white_rook`
6. `white_queen`
7. `white_king`
8. `black_pawn`
9. `black_knight`
10. `black_bishop`
11. `black_rook`
12. `black_queen`
13. `black_king`

### Expected Class Distribution
For a typical game:
- **Empty squares**: ~40-60% (varies by game stage)
- **Pawns**: Most common pieces (~20-30%)
- **Rooks, Knights, Bishops**: Medium frequency (~10-15%)
- **Queens, Kings**: Least common (~2-4%)

## 🎓 Development Roadmap

### ✅ Phase 0: Preprocessing (COMPLETED)
- [x] Implement board detection & warping
- [x] Implement square extraction
- [x] Implement FEN parsing
- [x] Process all labeled games
- [x] Create dataset splitting script
- [x] Create padded dataset variant

### ✅ Phase 1: Dataset Preparation (COMPLETED)
- [x] **Run `split_dataset.py` to create train/val/test splits**
  - CRITICAL: Splits by game to prevent data leakage
  - Output: `dataset/train/`, `dataset/val/`, `dataset/test/`
- [x] Verify splits are balanced

### ✅ Phase 2: Model Training (COMPLETED)
- [x] Implement model architectures (ResNet18, ResNet50, VGG16)
- [x] Implement training loop with early stopping
- [x] Implement data loading with class balancing
- [x] Add data augmentation
- [x] Integrate Comet.ml experiment tracking
- [x] Train baseline classifier (ResNet18)
- [x] Achieve target accuracy (~89% on validation)

### ✅ Phase 3: Evaluation & Analysis (COMPLETED)
- [x] Implement evaluation script with metrics
- [x] Generate confusion matrices
- [x] Per-class precision/recall/F1
- [x] Confidence distribution analysis
- [x] OOD detection using confidence thresholding
- [x] Visualize model predictions

### 🔴 Phase 4: Board Reconstruction (DO THIS NEXT!)
- [ ] Build full inference pipeline (image → FEN)
- [ ] Integrate classifier + OOD detector
- [ ] Implement FEN generation with '?' for unknowns
- [ ] Create visualization tools
- [ ] Test on sample images

### 📚 Phase 5: Advanced OOD Methods (Optional Enhancement)
- [ ] Implement ODIN (Out-of-Distribution detector)
- [ ] Implement Mahalanobis distance-based detection
- [ ] Compare with confidence-based method
- [ ] See [OOD_DETECTION_GUIDE.md](OOD_DETECTION_GUIDE.md) for details

### 📊 Phase 6: Final Evaluation & Report (1-2 days)
- [ ] Evaluate on test set (per-square metrics)
- [ ] Calculate board-level accuracy
- [ ] Perform comprehensive error analysis
- [ ] Test OOD detection on manually labeled occluded images
- [ ] Write final results report
- [ ] Generate all plots for presentation

### 🔄 Phase 7: Iteration & Polish (Ongoing)
- [ ] Identify failure modes
- [ ] Improve based on error analysis
- [ ] Consider using PGN data for additional training
- [ ] Test on new games for generalization
- [ ] Document findings and recommendations

**👉 See [GETTING_STARTED.md](GETTING_STARTED.md) for detailed breakdown of each phase**

## 🔍 Troubleshooting

### Board Detection Fails
**Symptoms**: Many images in `failed_detections/`

**Solutions**:
1. Check edge detection parameters in `BoardDetector._find_board_corners()`
2. Adjust Canny thresholds (currently 50, 150)
3. Try alternative detection method
4. Manually inspect failed images for patterns

### Poor Square Quality
**Symptoms**: Squares are misaligned or distorted

**Solutions**:
1. Verify corner detection is accurate (use `debug=True`)
2. Check that corners are ordered correctly
3. Ensure perspective transform is applied properly
4. Increase `board_size` for higher resolution

### Class Imbalance
**Symptoms**: Some classes have very few examples

**Solutions**:
1. Use data augmentation for rare classes
2. Apply class weights during training
3. Use focal loss or similar techniques
4. Generate synthetic training data

## 📚 Key Resources

### Computer Vision
- **OpenCV Perspective Transform**: [docs.opencv.org](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)
- **Contour Detection**: [docs.opencv.org/contours](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)

### Chess Notation
- **FEN Notation**: [Wikipedia](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation)
- **PGN Format**: [Wikipedia](https://en.wikipedia.org/wiki/Portable_Game_Notation)
- **python-chess**: [python-chess.readthedocs.io](https://python-chess.readthedocs.io/)

### Out-of-Distribution Detection
- **OOD Survey Paper**: [arXiv:2110.11334](https://arxiv.org/abs/2110.11334)
- **ODIN**: [arXiv:1706.02690](https://arxiv.org/abs/1706.02690)
- **Mahalanobis Distance**: [arXiv:1807.03888](https://arxiv.org/abs/1807.03888)

### Datasets
- **Roboflow Chess Datasets**: [universe.roboflow.com](https://universe.roboflow.com/)

## 🤝 Contributing

When extending this project:
1. Follow existing code structure and naming conventions
2. Add comments for complex/tricky logic
3. Update documentation (README, docstrings)
4. Test on sample data before full runs
5. Save intermediate results for debugging

## 📝 Notes

### Design Decisions

1. **512×512 board size**: Balance between quality and memory
2. **64×64 squares**: Large enough for detail, matches 8×8 grid
3. **Top-down perspective**: Eliminates angle distortion
4. **Class-based folders**: Standard format for image classification
5. **Traceable filenames**: Easy to debug and verify labels

### Tricky Parts

1. **Corner ordering**: Must be consistent (TL, TR, BR, BL)
2. **FEN parsing**: Numbers expand to multiple empties
3. **Square indexing**: Must match FEN order (rank 8→1, file a→h)
4. **Fallback detection**: Handles cases where edge detection fails
5. **Error handling**: Continue processing even if some frames fail

## 📄 License

This project is for educational purposes as part of the "Introduction to Deep Learning" course.

## 👥 Authors

BGU Deep Learning Course - Final Project

---

**📖 Documentation:**
- [GETTING_STARTED.md](GETTING_STARTED.md) - ⭐ Setup and detailed roadmap
- [OOD_DETECTION_GUIDE.md](OOD_DETECTION_GUIDE.md) - Guide to handling occlusions
- [preprocessing/README.md](preprocessing/README.md) - Module details and API
- [preprocessing/pipeline_diagram.txt](preprocessing/pipeline_diagram.txt) - Visual pipeline
- [preprocessing/IMPLEMENTATION_SUMMARY.md](preprocessing/IMPLEMENTATION_SUMMARY.md) - What was built
