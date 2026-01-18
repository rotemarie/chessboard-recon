# Chessboard Preprocessing Pipeline

Converts raw chessboard images into labeled 64×64 pixel squares for training.

## 📥 Get Data

Download from: https://drive.google.com/drive/folders/1WBEpr_TlmAv0hlVfa9ORQXABOIlqjwWz?usp=sharing

- `data/` - Raw images and FEN labels
- `preprocessed_data/` - Extracted squares (before split)
- `dataset/` - Train/val/test splits (70/15/15)

## 🔄 Pipeline Flow

```
Raw Image → Board Detection → Perspective Warp → Square Extraction → Labeled Squares
(angled)     (find corners)    (512×512 flat)    (64 × 64×64px)    (13 classes)
```

## 📁 Modules

### `board_detector.py` - Find and warp the board
```python
from preprocessing.board_detector import BoardDetector

detector = BoardDetector(board_size=512)
warped_board = detector.detect_board(image)  # Returns 512×512 top-down view
```

**How it works:**
1. Grayscale + Gaussian blur
2. Canny edge detection → find contours → select largest quadrilateral
3. Order corners: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
4. Apply perspective transform to 512×512 square
5. Fallback: Adaptive thresholding if edges fail

### `square_extractor.py` - Extract 64 squares + parse FEN
```python
from preprocessing.square_extractor import SquareExtractor, FENParser

# Extract squares
extractor = SquareExtractor(board_size=512)
squares = extractor.extract_squares(warped_board)  # Returns 64 images (64×64 each)

# Parse FEN to labels
labels = FENParser.fen_to_labels("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
# Returns: ['black_rook', 'black_knight', ..., 'empty', ...]  (64 labels)
```

**FEN Notation:**
- `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR` = starting position
- Uppercase = white pieces (R, N, B, Q, K, P)
- Lowercase = black pieces (r, n, b, q, k, p)
- Numbers = consecutive empty squares (8 = 8 empties)
- `/` = rank separator

### `preprocess_data.py` - Process entire dataset
```bash
python preprocess_data.py
```

Processes all games (~517 frames → ~30,000 squares) and organizes by class.

### `split_dataset.py` - Train/val/test split
```bash
python split_dataset.py
```

**CRITICAL:** Splits by **game** (not frame) to prevent data leakage!

## 📊 Output Structure

```
dataset/
├── train/              # 70%
│   ├── white_pawn/
│   ├── black_knight/
│   ├── empty/
│   └── ... (13 classes)
├── val/                # 15%
└── test/               # 15%
```

## 🎯 13 Piece Classes

1. `empty`
2-7. `white_pawn`, `white_knight`, `white_bishop`, `white_rook`, `white_queen`, `white_king`
8-13. `black_pawn`, `black_knight`, `black_bishop`, `black_rook`, `black_queen`, `black_king`

## 🧪 Test Pipeline

```bash
python test_pipeline.py
```

Runs 4 tests: board detection, square extraction, FEN parsing, full pipeline.

---

**Key Design Decisions:**
- 512×512 board size (power of 2, good quality)
- 64×64 squares (matches 8×8 grid)
- FEN ordering: rank 8→1, files a→h (index 0 = a8, 63 = h1)
- Filename format: `game2_frame_000200_a8.jpg` (traceable)
