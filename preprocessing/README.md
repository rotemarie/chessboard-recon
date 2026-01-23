# Chessboard Preprocessing Pipeline

This directory contains the preprocessing pipeline for converting raw chessboard images into labeled square images suitable for training a classifier.

## Get Data

The data can be found at this link: https://drive.google.com/drive/folders/1WBEpr_TlmAv0hlVfa9ORQXABOIlqjwWz?usp=sharing

This folder contains:
- **Processed dataset**: `dataset_blocks_black.zip` - Pre-processed 3×3 block crops, ready for training

## Overview

The preprocessing pipeline consists of three main steps:

1. **Board Detection & Warping**: Find the chessboard in the image and apply perspective transformation to get a top-down view (512×512 pixels)
2. **Block Extraction**: Extract 64 3×3 block crops (192×192 pixels each), centered on each square, providing context for classification
3. **Labeling**: Label each block based on the piece at the center square using FEN notation from the CSV files

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

### 2. `create_block_dataset.py`

Handles 3×3 block extraction and FEN parsing - **this is the current method used**.

**Key Features:**
- Extracts 64 blocks (3×3 squares, 192×192 pixels) centered on each target square
- Provides surrounding context to help distinguish similar pieces (e.g., bishops vs queens)
- Uses black padding (`cv2.BORDER_CONSTANT`) for edge blocks to avoid artifacts
- Maps block indices to chess notation (a8, b8, ..., h1)
- Parses FEN notation to get piece labels for each block's center square

**Main Classes:**
- `BlockSquareExtractor`: Extracts 3×3 block crops with configurable padding
- `FENParser`: Converts between FEN notation and piece labels (in `square_extractor.py`)

**Usage:**
```python
from create_block_dataset import BlockSquareExtractor
from square_extractor import FENParser

extractor = BlockSquareExtractor(
    board_size=512, 
    border_mode='constant', 
    border_color='black'
)
blocks = extractor.extract_blocks(warped_board)  # 64 blocks, each 192×192

labels = FENParser.fen_to_labels("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
```

**Why 3×3 Blocks?**
- **Problem**: Original 1×1 square crops (64×64) cut off tall pieces (queens, kings, bishops) when photographed at an angle
- **Solution**: 3×3 blocks capture the target square plus surrounding context, significantly improving accuracy on tall pieces
- **Result**: Validation accuracy improved from ~85% to ~89%, with dramatic improvements on queen/king/bishop classes

### 3. `square_extractor.py`

Original square extraction (legacy, kept for reference and FEN parsing utilities).

**Note**: The project now uses `BlockSquareExtractor` from `create_block_dataset.py` for better performance.

### 4. `preprocess_data.py`

Main preprocessing script (uses original square extraction, for reference).

**For block-based preprocessing**, run `create_block_dataset.py` directly (see its `main()` function).

### 5. `split_dataset.py`

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

### After Block Preprocessing (`create_block_dataset.py`)

```
preprocessed_data_blocks/
├── train/                      # All labeled 3×3 blocks (before split)
│   ├── black_bishop/
│   │   ├── game2_frame_000200_c8.jpg  # 192×192 block centered on c8
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
```

### After Splitting (`split_dataset.py`)

```
dataset_blocks/
├── train/                      # Training set (70% of games)
│   ├── black_bishop/           # 192×192 block crops
│   ├── black_king/
│   └── ... (all 13 classes)
├── val/                        # Validation set (15% of games)
│   ├── black_bishop/
│   ├── black_king/
│   └── ... (all 13 classes)
├── test/                       # Test set (15% of games)
│   ├── black_bishop/
│   ├── black_king/
│   └── ... (all 13 classes)
└── split_info.json            # Split metadata (reproducibility)
```

**Important:** The split is done BY GAME to prevent data leakage!

### Legacy Output (Original Square Extraction)

For reference, the original pipeline (`preprocess_data.py`) created 64×64 crops:
- `preprocessed_data/` - 64×64 square crops
- `dataset/` - Train/val/test splits of 64×64 crops

**Note**: The block-based approach (`dataset_blocks/`) is now the recommended method.

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



