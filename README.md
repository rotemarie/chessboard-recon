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
│   └── PGN/                        # Games with PGN files
│       ├── c06/ (game8, game9, game10)
│       └── c17/ (game11, game12, game13)
│
├── preprocessing/                  # Data preprocessing pipeline
│   ├── board_detector.py          # Board detection & warping
│   ├── square_extractor.py        # Square extraction & FEN parsing
│   ├── preprocess_data.py         # Main preprocessing script
│   ├── test_pipeline.py           # Testing script
│   ├── pipeline_diagram.txt       # Visual pipeline explanation
│   └── README.md                  # Preprocessing documentation
│
├── preprocessed_data/             # Processed data (created after running)
│   ├── train/                     # Labeled squares by class
│   │   ├── white_pawn/
│   │   ├── black_knight/
│   │   ├── empty/
│   │   └── ... (13 classes total)
│   ├── warped_boards/             # Warped boards for inspection
│   ├── failed_detections/         # Failed cases for debugging
│   └── metadata/                  # Processing logs
│
├── requirements.txt               # Python dependencies
├── SETUP.md                       # Installation guide
└── README.md                      # This file
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

## 🎓 Next Steps - Development Roadmap

### ✅ Phase 0: Preprocessing (DONE)
- [x] Implement board detection & warping
- [x] Implement square extraction
- [x] Implement FEN parsing
- [x] Process all labeled games
- [x] Create dataset splitting script

### 🔴 Phase 1: Dataset Preparation (DO THIS NEXT!)
- [ ] **Run `split_dataset.py` to create train/val/test splits**
  - CRITICAL: Splits by game to prevent data leakage
  - Output: `dataset/train/`, `dataset/val/`, `dataset/test/`
- [ ] Verify splits are balanced

### 📚 Phase 2: Research & Design (1-2 days)
- [ ] **Research OOD detection methods** (ODIN, Mahalanobis, Energy-based)
  - This is a CORE requirement, not optional!
  - Must handle occluded squares
- [ ] Choose model architecture (ResNet, EfficientNet, MobileNet)
- [ ] Design data augmentation strategy
  - Include simulated occlusions for OOD training

### 💻 Phase 3: Implementation (3-5 days)
- [ ] Implement dataset and dataloaders
- [ ] Implement model architecture with OOD support
- [ ] Implement training loop with proper metrics
- [ ] Train baseline classifier
- [ ] Implement and tune OOD detector
- [ ] Achieve target accuracy (>85% on validation)

### 🔗 Phase 4: Board Reconstruction (2-3 days)
- [ ] Build full inference pipeline (image → FEN)
- [ ] Integrate classifier + OOD detector
- [ ] Implement FEN generation with '?' for unknowns
- [ ] Create visualization tools
- [ ] Test on sample images

### 📊 Phase 5: Evaluation (1-2 days)
- [ ] Evaluate on test set (per-square metrics)
- [ ] Calculate board-level accuracy
- [ ] Perform error analysis
- [ ] Test OOD detection on occluded images
- [ ] Write results report

### 🔄 Phase 6: Iteration (Ongoing)
- [ ] Identify failure modes
- [ ] Improve based on error analysis
- [ ] Consider using PGN data for additional training
- [ ] Test on new games for generalization

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
