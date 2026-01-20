"""
Create Block-Based Square Dataset

This script creates a dataset where each square is extracted as the center of a 3×3 block
of squares. This provides context from neighboring squares to help with classification.

For each of the 64 squares:
- Extract a 3×3 block centered on that square
- The target square is in the center
- Output: 64 images, each 192×192 pixels (3 × 64px squares)

Edge handling:
- For squares on the edge/corner, pad with the board border (replicated pixels)
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

try:
    from .board_detector import BoardDetector
    from .square_extractor import FENParser
except ImportError:
    from board_detector import BoardDetector
    from square_extractor import FENParser


class BlockSquareExtractor:
    """
    Extracts 3×3 blocks centered on each square.
    """
    
    def __init__(
        self,
        board_size: int = 512,
        border_mode: str = 'constant',
        border_color: str = 'black',
        verbose: bool = True,
    ):
        """
        Initialize the block extractor.
        
        Args:
            board_size: Size of the warped board (must match BoardDetector)
            border_mode: How to handle edges - 'reflect', 'constant', or 'replicate'
            border_color: Color for constant border - 'black', 'white', or 'gray'
        """
        self.board_size = board_size
        self.square_size = board_size // 8  # 64 pixels per square
        self.block_size = 3  # 3×3 blocks
        self.output_size = self.square_size * self.block_size  # 192×192
        
        # Choose border mode
        self.border_modes = {
            'reflect': cv2.BORDER_REFLECT_101,  # Mirror reflection (no edge duplication)
            'constant': cv2.BORDER_CONSTANT,    # Fill with constant color
            'replicate': cv2.BORDER_REPLICATE   # Replicate edge pixels (causes blur)
        }
        self.border_mode = self.border_modes.get(border_mode, cv2.BORDER_CONSTANT)
        self.border_mode_name = border_mode
        
        # Choose border color (BGR format)
        self.border_colors = {
            'black': [0, 0, 0],
            'white': [255, 255, 255],
            'gray': [128, 128, 128]
        }
        self.border_color = self.border_colors.get(border_color, [0, 0, 0])
        self.border_color_name = border_color
        
        if verbose:
            print(f"Square size: {self.square_size}×{self.square_size}")
            print(f"Block size: {self.block_size}×{self.block_size} squares")
            print(f"Output size: {self.output_size}×{self.output_size} pixels")
            print(f"Border handling: {border_mode}")
            if border_mode == 'constant':
                print(f"Border color: {border_color}")
    
    def extract_blocks(self, warped_board: np.ndarray) -> list:
        """
        Extract 64 blocks (one 3×3 block per square).
        
        Args:
            warped_board: Top-down view of the board (512×512)
            
        Returns:
            List of 64 block images (192×192 each)
        """
        # Pad the board by 1 square on each side for edge handling
        # This allows us to extract 3×3 blocks for edge squares
        if self.border_mode == cv2.BORDER_CONSTANT:
            # Use specified color for constant border
            padded_board = cv2.copyMakeBorder(
                warped_board,
                self.square_size,  # top
                self.square_size,  # bottom
                self.square_size,  # left
                self.square_size,  # right
                self.border_mode,
                value=self.border_color
            )
        else:
            # Use the specified border mode
            padded_board = cv2.copyMakeBorder(
                warped_board,
                self.square_size,  # top
                self.square_size,  # bottom
                self.square_size,  # left
                self.square_size,  # right
                self.border_mode
            )
        
        blocks = []
        
        # For each square on the board (8×8 = 64)
        for row in range(8):
            for col in range(8):
                # Calculate center of target square in padded board
                # (adding 1 because of padding)
                center_row = row + 1
                center_col = col + 1
                
                # Extract 3×3 block centered on this square
                # Start 1 square before center
                y1 = (center_row - 1) * self.square_size
                y2 = (center_row + 2) * self.square_size  # +2 to include 3 squares
                x1 = (center_col - 1) * self.square_size
                x2 = (center_col + 2) * self.square_size
                
                # Extract block
                block = padded_board[y1:y2, x1:x2]
                blocks.append(block)
        
        return blocks
    
    def get_square_position(self, square_idx: int) -> str:
        """
        Get chess notation for a square index.
        
        Args:
            square_idx: Square index (0-63)
            
        Returns:
            Chess notation (e.g., "a8", "h1")
        """
        row = square_idx // 8
        col = square_idx % 8
        file = chr(ord('a') + col)
        rank = 8 - row
        return f"{file}{rank}"
    
    def visualize_block(self, warped_board: np.ndarray, square_idx: int):
        """
        Visualize a single 3×3 block with the center square highlighted.
        
        Args:
            warped_board: Warped board image
            square_idx: Which square to visualize (0-63)
        """
        import matplotlib.pyplot as plt
        
        blocks = self.extract_blocks(warped_board)
        block = blocks[square_idx]
        
        # Draw a rectangle around the center square
        block_copy = block.copy()
        center_start = self.square_size  # Center square starts at pixel 64
        center_end = 2 * self.square_size  # Center square ends at pixel 128
        
        # Draw red rectangle around center
        cv2.rectangle(
            block_copy,
            (center_start, center_start),
            (center_end, center_end),
            (0, 0, 255),  # Red
            3  # Thickness
        )
        
        # Visualize
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(block_copy, cv2.COLOR_BGR2RGB))
        plt.title(f'3×3 Block for square {self.get_square_position(square_idx)}\n'
                 f'(Center square highlighted in red)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def create_block_dataset(data_root: str, 
                        output_root: str,
                        board_size: int = 512,
                        border_mode: str = 'constant',
                        border_color: str = 'black'):
    """
    Create a block-based dataset where each square is extracted with 3×3 context.
    
    Args:
        data_root: Root with per_frame data
        output_root: Where to save block dataset
        board_size: Board size for warping
        border_mode: How to handle edges - 'reflect' (mirror), 'constant' (solid color), or 'replicate' (blur)
        border_color: Color for constant border - 'black', 'white', or 'gray'
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    
    print("="*70)
    print("CREATING BLOCK-BASED DATASET (3×3 blocks)")
    print("="*70)
    print(f"Input: {data_root}")
    print(f"Output: {output_root}")
    print(f"Border mode: {border_mode}")
    if border_mode == 'constant':
        print(f"Border color: {border_color}")
    print()
    
    # Initialize
    detector = BoardDetector(board_size=board_size)
    extractor = BlockSquareExtractor(board_size=board_size, border_mode=border_mode, border_color=border_color)
    
    # Create output structure
    train_dir = output_root / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)
    
    for class_name in FENParser.get_piece_classes():
        (train_dir / class_name).mkdir(exist_ok=True)
    
    # Process each game
    per_frame_dir = data_root / 'per_frame'
    game_dirs = [d for d in per_frame_dir.iterdir() 
                if d.is_dir() and 'game' in d.name and '_per_frame' in d.name]
    
    total_stats = {'success': 0, 'failed': 0, 'total_blocks': 0}
    
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
            
            # Extract 3×3 blocks
            blocks = extractor.extract_blocks(warped)
            
            # Get labels
            try:
                labels = FENParser.fen_to_labels(fen)
            except ValueError:
                total_stats['failed'] += 1
                continue
            
            # Save each block (one per square)
            for square_idx, (block, label) in enumerate(zip(blocks, labels)):
                position = extractor.get_square_position(square_idx)
                filename = f"{game_name}_frame_{frame_num:06d}_{position}.jpg"
                output_path = train_dir / label / filename
                cv2.imwrite(str(output_path), block)
                total_stats['total_blocks'] += 1
            
            total_stats['success'] += 1
    
    # Print statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    print(f"Frames processed: {total_stats['success']}")
    print(f"Frames failed: {total_stats['failed']}")
    print(f"Total 3×3 blocks: {total_stats['total_blocks']}")
    print(f"\nOutput directory: {output_root}")
    
    # Print class distribution
    print("\n" + "="*70)
    print("CLASS DISTRIBUTION")
    print("="*70)
    for class_dir in sorted(train_dir.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.jpg')))
            print(f"  {class_dir.name:20s}: {count:6d} images")


def test_blocks(border_mode: str = 'constant', border_color: str = 'black'):
    """
    Test block extraction on a sample image.
    
    Args:
        border_mode: Border handling strategy to test
        border_color: Color for constant border
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
    
    # Extract blocks with specified border mode
    extractor = BlockSquareExtractor(board_size=512, border_mode=border_mode, border_color=border_color)
    
    # Visualize some interesting squares
    # e4 (center) = square 28
    # a8 (corner) = square 0
    # h1 (corner) = square 63
    print("\nVisualizing 3×3 blocks:")
    print("- Square a8 (corner)")
    extractor.visualize_block(warped, square_idx=0)
    
    print("- Square e4 (center)")
    extractor.visualize_block(warped, square_idx=28)
    
    print("- Square h1 (corner)")
    extractor.visualize_block(warped, square_idx=63)


def main():
    """
    Main entry point.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Create 3×3 block-based dataset')
    parser.add_argument('--data-root', type=str,
                       default='/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon/data',
                       help='Root directory with per_frame data')
    parser.add_argument('--output-root', type=str,
                       default='/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon/preprocessed_data_blocks',
                       help='Output directory for block dataset')
    parser.add_argument('--border-mode', type=str, 
                       default='constant',
                       choices=['reflect', 'constant', 'replicate'],
                       help='Border handling: reflect (mirror), constant (solid color), replicate (blur)')
    parser.add_argument('--border-color', type=str,
                       default='black',
                       choices=['black', 'white', 'gray'],
                       help='Color for constant border: black, white, or gray')
    parser.add_argument('--test', action='store_true',
                       help='Run test visualization instead of full processing')
    
    args = parser.parse_args()
    
    if args.test:
        print("Running test visualization...")
        test_blocks(border_mode=args.border_mode, border_color=args.border_color)
    else:
        create_block_dataset(
            data_root=args.data_root,
            output_root=args.output_root,
            border_mode=args.border_mode,
            border_color=args.border_color
        )
        
        print("\n✅ Block-based dataset created successfully!")
        print("\nNext steps:")
        print("  1. Run split_dataset.py on the block data")
        print("  2. Train models with 3×3 block context")
        print("  3. Compare performance with single-square models")


if __name__ == "__main__":
    main()

