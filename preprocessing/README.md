# Chessboard Preprocessing Pipeline

This directory contains the preprocessing pipeline for converting raw chessboard images into labeled square images suitable for training a classifier.

## Get Data

The data can be found at this link: https://drive.google.com/drive/folders/1WBEpr_TlmAv0hlVfa9ORQXABOIlqjwWz?usp=sharing
- data/ - Raw chessboard images and labels
- preprocessed_data/ - All labeled squares extracted from raw images (before split)
- dataset/ - Split data organized into train/val/test sets (70/15/15)

## Overview

The preprocessing pipeline consists of three main steps:

1. **Board Detection & Warping**: Find the chessboard in the image and apply perspective transformation to get a top-down view
2. **Square Extraction**: Slice the warped board into 64 individual square images
3. **Labeling**: Label each square based on the FEN notation from the CSV files

## Modules

### 1. `board_detector.py`

Handles board detection and perspective transformation.

**Key Features:**
- Uses edge detection and contour finding to locate the board
- Falls back to adaptive thresholding if edge detection fails
- Orders corners correctly (top-left, top-right, bottom-right, bottom-left)
- Applies perspective transform to create a square, top-down view

**Main Class:** `BoardDetector`

**Usage:**
```python
from board_detector import BoardDetector

detector = BoardDetector(board_size=512)
warped = detector.detect_board(image, debug=False)
```

### 2. `square_extractor.py`

Handles square extraction and FEN parsing.

**Key Features:**
- Extracts 64 squares from a warped board
- Maps square indices to chess notation (a8, b8, ..., h1)
- Parses FEN notation to get piece labels for each square
- Supports visualization of extracted squares

**Main Classes:**
- `SquareExtractor`: Extracts and visualizes squares
- `FENParser`: Converts between FEN notation and piece labels

**Usage:**
```python
from square_extractor import SquareExtractor, FENParser

extractor = SquareExtractor(board_size=512)
squares = extractor.extract_squares(warped_board)

labels = FENParser.fen_to_labels("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
```

### 3. `preprocess_data.py`

Main preprocessing script that ties everything together.

### 4. `split_dataset.py`

Dataset splitting script that creates train/val/test splits.

**Key Features:**
- Processes all games in the `per_frame` directory
- Saves extracted squares organized by piece class
- Tracks processing statistics and saves metadata
- Handles errors gracefully (saves failed detections for inspection)

**Main Class:** `ChessDataPreprocessor`

**Usage:**
```bash
python preprocess_data.py
```

---

**Main Class:** `DatasetSplitter`

**Usage:**
```bash
python split_dataset.py
```

**CRITICAL:** This script splits by GAME, not by frame, to prevent data leakage. Consecutive frames from the same game are highly correlated and would lead to artificially inflated validation/test performance.

## Output Structure

### After Preprocessing (`preprocess_data.py`)

```
preprocessed_data/
├── train/                      # All labeled squares (before split)
│   ├── black_bishop/
│   │   ├── game2_frame_000200_c8.jpg
│   │   ├── game2_frame_000200_f8.jpg
│   │   └── ...
│   ├── black_king/
│   ├── black_knight/
│   ├── black_pawn/
│   ├── black_queen/
│   ├── black_rook/
│   ├── white_bishop/
│   ├── white_king/
│   ├── white_knight/
│   ├── white_pawn/
│   ├── white_queen/
│   ├── white_rook/
│   └── empty/
├── warped_boards/              # For inspection
│   ├── game2_frame_000200.jpg
│   └── ...
├── failed_detections/          # Failed board detections
│   └── ...
└── metadata/                   # Processing logs
    ├── game2_metadata.csv
    └── ...
```

### After Splitting (`split_dataset.py`)

```
dataset/
├── train/                      # Training set (70%)
│   ├── black_bishop/
│   ├── black_king/
│   └── ... (all 13 classes)
├── val/                        # Validation set (15%)
│   ├── black_bishop/
│   ├── black_king/
│   └── ... (all 13 classes)
├── test/                       # Test set (15%)
│   ├── black_bishop/
│   ├── black_king/
│   └── ... (all 13 classes)
└── split_info.json            # Split metadata (reproducibility)
```

**Important:** The split is done BY GAME to prevent data leakage!

## Piece Classes

The system recognizes 13 classes:
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

## FEN Notation

FEN (Forsyth-Edwards Notation) describes the board state:
- Board is described rank by rank (8 to 1), file by file (a to h)
- Uppercase letters = white pieces (P, N, B, R, Q, K)
- Lowercase letters = black pieces (p, n, b, r, q, k)
- Numbers = consecutive empty squares
- `/` = separates ranks

Example: `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR` (starting position)

## Testing Individual Modules

Each module can be tested independently:

```bash
# Test board detection
python board_detector.py

# Test square extraction
python square_extractor.py
```



