"""
Create padded square dataset from game videos (games 2-7 only).
Each square is extracted with configurable padding (default 30%) to provide context.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import sys

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.board_detector import BoardDetector


class PaddedSquareExtractor:
    """Extract squares with padding from a warped board."""
    
    def __init__(self, board_size: int = 512, padding_percent: float = 0.30):
        """
        Args:
            board_size: Size of the warped board (default 512x512)
            padding_percent: Percentage of square size to add as padding (e.g., 0.30 for 30%)
        """
        self.board_size = board_size
        self.square_size = board_size // 8
        self.padding_percent = padding_percent
        self.padding_pixels = int(self.square_size * padding_percent)
        
    def extract_squares_with_padding(self, warped_board: np.ndarray) -> list:
        """
        Extract 64 squares with padding from the warped board.
        Uses black border padding.
        
        Returns:
            List of 64 squares (from a8 to h1) with padding
        """
        # Add padding to the entire board with black border
        padded_board = cv2.copyMakeBorder(
            warped_board,
            self.padding_pixels, self.padding_pixels,
            self.padding_pixels, self.padding_pixels,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # Black padding
        )
        
        squares = []
        # Extract squares in chess notation order (a8 to h1)
        for row in range(8):  # rows 8 to 1
            for col in range(8):  # columns a to h
                # Calculate position in the padded board
                y_start = row * self.square_size
                x_start = col * self.square_size
                
                # Extract square with padding (already in padded coordinates)
                square = padded_board[
                    y_start : y_start + self.square_size + 2 * self.padding_pixels,
                    x_start : x_start + self.square_size + 2 * self.padding_pixels
                ]
                
                squares.append(square)
        
        return squares


def get_square_name(square_idx: int) -> str:
    """Convert square index (0-63) to chess notation (a8-h1)."""
    row = square_idx // 8
    col = square_idx % 8
    file = chr(ord('a') + col)
    rank = 8 - row
    return f"{file}{rank}"


def process_game(game_dir: Path, output_dir: Path, detector: BoardDetector, 
                 extractor: PaddedSquareExtractor):
    """
    Process a single game: detect board, extract padded squares, save with labels.
    
    Args:
        game_dir: Path to game directory (e.g., data/per_frame/game2_per_frame)
        output_dir: Root output directory (preprocessed_data_padded/train)
        detector: BoardDetector instance
        extractor: PaddedSquareExtractor instance
    """
    game_name = game_dir.name.replace('_per_frame', '')
    csv_file = game_dir / f"{game_name}.csv"
    frames_dir = game_dir / "tagged_images"
    
    if not csv_file.exists() or not frames_dir.exists():
        print(f"⚠️  Skipping {game_name}: Missing CSV or frames directory")
        return
    
    # Read FEN labels
    df = pd.read_csv(csv_file)
    failed_count = 0
    success_count = 0
    
    print(f"\n📸 Processing {game_name}...")
    
    # Process each FEN position (from_frame to to_frame range)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {game_name}"):
        # Use from_frame as the representative frame
        frame_num = row['from_frame']
        frame_file = frames_dir / f"frame_{frame_num:06d}.jpg"
        
        if not frame_file.exists():
            failed_count += 1
            continue
        
        # Read frame
        frame = cv2.imread(str(frame_file))
        if frame is None:
            failed_count += 1
            continue
        
        # Detect board
        warped = detector.detect_board(frame, debug=False)
        if warped is None:
            failed_count += 1
            continue
        
        # Extract padded squares
        squares = extractor.extract_squares_with_padding(warped)
        
        # Parse FEN to get piece labels for each square
        fen = row['fen'].split()[0]  # Get board position part only
        board_str = fen.replace('/', '')
        
        # Expand FEN notation (e.g., "3" becomes "---")
        expanded_board = ""
        for char in board_str:
            if char.isdigit():
                expanded_board += '-' * int(char)
            else:
                expanded_board += char
        
        # Map FEN characters to class names
        piece_map = {
            'r': 'black_rook', 'n': 'black_knight', 'b': 'black_bishop',
            'q': 'black_queen', 'k': 'black_king', 'p': 'black_pawn',
            'R': 'white_rook', 'N': 'white_knight', 'B': 'white_bishop',
            'Q': 'white_queen', 'K': 'white_king', 'P': 'white_pawn',
            '-': 'empty'
        }
        
        # Save each square to its class directory
        frame_name = frame_file.stem
        for idx, square in enumerate(squares):
            piece = expanded_board[idx]
            class_name = piece_map.get(piece, 'empty')
            square_name = get_square_name(idx)
            
            # Create class directory
            class_dir = output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Save square image
            output_file = class_dir / f"{game_name}_{frame_name}_{square_name}.jpg"
            cv2.imwrite(str(output_file), square)
        
        success_count += 1
    
    print(f"  ✅ {success_count} frames processed, {failed_count} failed")


def main():
    """Process all tagged games (2-7) with padded square extraction."""
    
    # Configuration
    DATA_ROOT = Path("/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon/data/per_frame")
    OUTPUT_ROOT = Path("/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon/preprocessed_data_padded")
    BOARD_SIZE = 512
    PADDING_PERCENT = 0.30  # 30% padding
    
    # Only process manually tagged games (2, 4-7)
    GAMES_TO_PROCESS = ['game2_per_frame', 'game4_per_frame', 'game5_per_frame', 'game6_per_frame', 'game7_per_frame']
    
    # Create output directory
    train_dir = OUTPUT_ROOT / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector and extractor
    detector = BoardDetector(board_size=BOARD_SIZE)
    extractor = PaddedSquareExtractor(board_size=BOARD_SIZE, padding_percent=PADDING_PERCENT)
    
    print(f"🎯 Padded Square Extraction")
    print(f"   Board size: {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"   Padding: {PADDING_PERCENT*100:.0f}% ({extractor.padding_pixels}px)")
    print(f"   Final square size: {BOARD_SIZE//8 + 2*extractor.padding_pixels}x{BOARD_SIZE//8 + 2*extractor.padding_pixels}")
    print(f"   Output: {OUTPUT_ROOT}")
    print(f"   Games: {', '.join(GAMES_TO_PROCESS)}")
    
    # Process each game
    for game_name in GAMES_TO_PROCESS:
        game_dir = DATA_ROOT / game_name
        if not game_dir.exists():
            print(f"⚠️  Warning: {game_name} directory not found, skipping")
            continue
        process_game(game_dir, train_dir, detector, extractor)
    
    # Summary
    print("\n📊 Dataset Summary:")
    total_images = 0
    for class_dir in sorted(train_dir.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.jpg')))
            total_images += count
            print(f"   {class_dir.name}: {count} images")
    print(f"\n   ✅ Total: {total_images} images")
    print(f"   📁 Saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
