"""
Create Padded Square Dataset

This script creates an alternative dataset where each square has padding
around it to capture pieces that extend beyond square boundaries (especially
at angles or tall pieces like kings/queens).

The padding helps capture:
- Piece "heads" that stick out beyond square boundaries
- Shadows and context from neighboring squares
- Better features for angled pieces

Default: 30% padding with solid black borders (no blur artifacts)
Output: preprocessed_data_padded/ with larger square images (102×102 for 30%)
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from board_detector import BoardDetector
from square_extractor import FENParser


class PaddedSquareExtractor:
    """
    Extracts squares with padding around them.
    """
    
    def __init__(self, board_size: int = 512, padding_percent: float = 0.30, border_color: str = 'black'):
        """
        Initialize the padded square extractor.
        
        Args:
            board_size: Size of the warped board (must match BoardDetector)
            padding_percent: Percentage of padding to add (0.30 = 30% on each side)
                           For 64x64 square, 30% padding = 102x102 output
            border_color: Color for padding - 'black', 'white', or 'gray'
        """
        self.board_size = board_size
        self.square_size = board_size // 8
        self.padding_percent = padding_percent
        self.padding_pixels = int(self.square_size * padding_percent)
        self.output_size = self.square_size + 2 * self.padding_pixels
        
        # Choose border color (BGR format)
        self.border_colors = {
            'black': [0, 0, 0],
            'white': [255, 255, 255],
            'gray': [128, 128, 128]
        }
        self.border_color = self.border_colors.get(border_color, [0, 0, 0])
        self.border_color_name = border_color
        
        print(f"Square size: {self.square_size}x{self.square_size}")
        print(f"Padding: {self.padding_pixels} pixels on each side")
        print(f"Output size: {self.output_size}x{self.output_size}")
        print(f"Border color: {border_color}")
    
    def extract_squares_with_padding(self, warped_board: np.ndarray) -> list:
        """
        Extract all 64 squares with padding from a warped board.
        
        Args:
            warped_board: Top-down view of the board (square image)
            
        Returns:
            List of 64 padded square images
        """
        # Add padding to the entire board first
        # This handles edge squares cleanly with solid color padding
        padded_board = cv2.copyMakeBorder(
            warped_board,
            self.padding_pixels,  # top
            self.padding_pixels,  # bottom
            self.padding_pixels,  # left
            self.padding_pixels,  # right
            cv2.BORDER_CONSTANT,  # Fill with constant color
            value=self.border_color
        )
        
        squares = []
        
        # Extract squares from padded board
        # Each extraction includes padding around the original square
        for row in range(8):
            for col in range(8):
                # Calculate coordinates in padded board
                # Start at original position (accounting for board padding)
                y1 = row * self.square_size
                y2 = y1 + self.output_size
                x1 = col * self.square_size
                x2 = x1 + self.output_size
                
                # Extract padded square
                square = padded_board[y1:y2, x1:x2]
                squares.append(square)
        
        return squares
    
    def get_square_position(self, square_idx: int) -> str:
        """
        Get chess notation for a square index.
        Same as SquareExtractor.
        """
        row = square_idx // 8
        col = square_idx % 8
        file = chr(ord('a') + col)
        rank = 8 - row
        return f"{file}{rank}"
    
    def visualize_comparison(self, warped_board: np.ndarray, square_idx: int = 0):
        """
        Compare normal vs padded extraction for a single square.
        
        Args:
            warped_board: Warped board image
            square_idx: Which square to visualize (0-63)
        """
        import matplotlib.pyplot as plt
        
        # Extract without padding
        row = square_idx // 8
        col = square_idx % 8
        y1 = row * self.square_size
        y2 = (row + 1) * self.square_size
        x1 = col * self.square_size
        x2 = (col + 1) * self.square_size
        normal_square = warped_board[y1:y2, x1:x2]
        
        # Extract with padding
        padded_squares = self.extract_squares_with_padding(warped_board)
        padded_square = padded_squares[square_idx]
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(cv2.cvtColor(normal_square, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Normal: {self.square_size}x{self.square_size}')
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(padded_square, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Padded: {self.output_size}x{self.output_size}')
        axes[1].axis('off')
        
        plt.suptitle(f'Square {self.get_square_position(square_idx)} Comparison')
        plt.tight_layout()
        plt.show()


def create_padded_dataset(data_root: str, 
                         output_root: str,
                         padding_percent: float = 0.30,
                         board_size: int = 512,
                         border_color: str = 'black'):
    """
    Create a padded version of the preprocessed dataset.
    
    Args:
        data_root: Root with per_frame data
        output_root: Where to save padded dataset
        padding_percent: How much padding (0.30 = 30%)
        board_size: Board size for warping
        border_color: Color for padding - 'black', 'white', or 'gray'
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    
    print("="*70)
    print("CREATING PADDED DATASET")
    print("="*70)
    print(f"Input: {data_root}")
    print(f"Output: {output_root}")
    print(f"Padding: {padding_percent*100:.0f}%")
    print(f"Border color: {border_color}")
    print()
    
    # Initialize
    detector = BoardDetector(board_size=board_size)
    extractor = PaddedSquareExtractor(board_size=board_size, 
                                     padding_percent=padding_percent,
                                     border_color=border_color)
    
    # Create output structure
    train_dir = output_root / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)
    
    for class_name in FENParser.get_piece_classes():
        (train_dir / class_name).mkdir(exist_ok=True)
    
    # Process each game
    per_frame_dir = data_root / 'per_frame'
    game_dirs = [d for d in per_frame_dir.iterdir() 
                if d.is_dir() and 'game' in d.name and '_per_frame' in d.name]
    
    total_stats = {'success': 0, 'failed': 0, 'total_squares': 0}
    
    for game_dir in sorted(game_dirs):
        game_name = game_dir.name.replace('_per_frame', '')
        print(f"\nProcessing {game_name}...")
        
        # Load CSV
        csv_file = game_dir / f"{game_name}.csv"
        if not csv_file.exists():
            print(f"  CSV not found, skipping")
            continue
        
        df = pd.read_csv(csv_file)
        images_dir = game_dir / 'tagged_images'
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {game_name}"):
            frame_num = row['from_frame']
            fen = row['fen']
            
            # Load image
            image_path = images_dir / f"frame_{frame_num:06d}.jpg"
            if not image_path.exists():
                total_stats['failed'] += 1
                continue
            
            image = cv2.imread(str(image_path))
            if image is None:
                total_stats['failed'] += 1
                continue
            
            # Detect and warp
            warped = detector.detect_board(image, debug=False)
            if warped is None:
                total_stats['failed'] += 1
                continue
            
            # Extract padded squares
            squares = extractor.extract_squares_with_padding(warped)
            
            # Get labels
            try:
                labels = FENParser.fen_to_labels(fen)
            except ValueError:
                total_stats['failed'] += 1
                continue
            
            # Save each square
            for square_idx, (square, label) in enumerate(zip(squares, labels)):
                position = extractor.get_square_position(square_idx)
                filename = f"{game_name}_frame_{frame_num:06d}_{position}.jpg"
                output_path = train_dir / label / filename
                cv2.imwrite(str(output_path), square)
                total_stats['total_squares'] += 1
            
            total_stats['success'] += 1
    
    # Print statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    print(f"Frames processed: {total_stats['success']}")
    print(f"Frames failed: {total_stats['failed']}")
    print(f"Total padded squares: {total_stats['total_squares']}")
    print(f"\nOutput directory: {output_root}")
    
    # Print class distribution
    print("\n" + "="*70)
    print("CLASS DISTRIBUTION")
    print("="*70)
    for class_dir in sorted(train_dir.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.jpg')))
            print(f"  {class_dir.name:20s}: {count:6d} images")


def test_padding():
    """
    Test padded extraction on a sample image.
    """
    from pathlib import Path
    
    # Load sample
    data_root = Path("/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon/data")
    sample_image = data_root / "per_frame/game2_per_frame/tagged_images/frame_000200.jpg"
    
    if not sample_image.exists():
        print("Sample image not found")
        return
    
    # Process
    image = cv2.imread(str(sample_image))
    detector = BoardDetector(board_size=512)
    warped = detector.detect_board(image, debug=False)
    
    if warped is None:
        print("Board detection failed")
        return
    
    # Compare padding percentages
    for padding in [0.20, 0.25, 0.30]:
        print(f"\nPadding: {padding*100:.0f}%")
        extractor = PaddedSquareExtractor(board_size=512, padding_percent=padding, border_color='black')
        
        # Show comparison for a piece square (e.g., white king at e1)
        # e1 = square 60
        extractor.visualize_comparison(warped, square_idx=60)


def main():
    """
    Main entry point.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Create padded square dataset')
    parser.add_argument('--data-root', type=str,
                       default='/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon/data',
                       help='Root directory with per_frame data')
    parser.add_argument('--output-root', type=str,
                       default='/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon/preprocessed_data_padded',
                       help='Output directory for padded dataset')
    parser.add_argument('--padding', type=float, default=0.30,
                       help='Padding percentage (0.30 = 30%)')
    parser.add_argument('--border-color', type=str, default='black',
                       choices=['black', 'white', 'gray'],
                       help='Color for padding: black, white, or gray')
    parser.add_argument('--test', action='store_true',
                       help='Run test visualization instead of full processing')
    
    args = parser.parse_args()
    
    if args.test:
        print("Running test visualization...")
        test_padding()
    else:
        create_padded_dataset(
            data_root=args.data_root,
            output_root=args.output_root,
            padding_percent=args.padding,
            border_color=args.border_color
        )
        
        print("\n✅ Padded dataset created successfully!")
        print("\nNext steps:")
        print("  1. Run split_dataset.py on the padded data")
        print("  2. Train models on both datasets")
        print("  3. Compare performance")


if __name__ == "__main__":
    main()

