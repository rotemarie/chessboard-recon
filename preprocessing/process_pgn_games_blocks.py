"""
Process PGN Games for Block Dataset

This script processes games from the PGN folder structure and adds them
to the existing block dataset. It extracts 3×3 blocks (192×192 pixels)
centered on each square to capture context from neighboring squares.

Usage:
    python process_pgn_games_blocks.py --output-root ../preprocessed_data_blocks
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import chess
import chess.pgn
import io
from typing import List, Tuple, Dict
import re

from board_detector import BoardDetector
from square_extractor import FENParser


class BlockSquareExtractor:
    """
    Extracts 3×3 blocks centered on each square.
    Each block is 192×192 pixels (3 squares × 64 pixels).
    """
    
    def __init__(self, board_size: int = 512, border_color: str = 'black'):
        """
        Initialize the block square extractor.
        
        Args:
            board_size: Size of the warped board (must match BoardDetector)
            border_color: Color for padding edges - 'black', 'white', or 'gray'
        """
        self.board_size = board_size
        self.square_size = board_size // 8  # 64 pixels
        self.block_size = self.square_size * 3  # 192 pixels (3×3 squares)
        
        # Border color (BGR format)
        self.border_colors = {
            'black': [0, 0, 0],
            'white': [255, 255, 255],
            'gray': [128, 128, 128]
        }
        self.border_color = self.border_colors.get(border_color, [0, 0, 0])
    
    def extract_blocks(self, warped_board: np.ndarray) -> list:
        """
        Extract all 64 blocks (3×3 centered on each square) from a warped board.
        
        Args:
            warped_board: Top-down view of the board (512×512)
            
        Returns:
            List of 64 block images (192×192 each)
        """
        # Pad the board by one square on each side for edge cases
        padded_board = cv2.copyMakeBorder(
            warped_board,
            self.square_size,  # top
            self.square_size,  # bottom
            self.square_size,  # left
            self.square_size,  # right
            cv2.BORDER_CONSTANT,
            value=self.border_color
        )
        
        blocks = []
        
        # Extract 3×3 block centered on each square
        for row in range(8):
            for col in range(8):
                # In padded coordinates, the original square starts at (square_size, square_size)
                # Center of the target square in padded coords
                center_y = self.square_size + row * self.square_size + self.square_size // 2
                center_x = self.square_size + col * self.square_size + self.square_size // 2
                
                # Block boundaries (1.5 squares in each direction from center)
                y1 = center_y - (3 * self.square_size) // 2
                y2 = y1 + self.block_size
                x1 = center_x - (3 * self.square_size) // 2
                x2 = x1 + self.block_size
                
                block = padded_board[y1:y2, x1:x2]
                blocks.append(block)
        
        return blocks
    
    def get_square_position(self, square_idx: int) -> str:
        """Get chess notation for a square index."""
        row = square_idx // 8
        col = square_idx % 8
        file = chr(ord('a') + col)
        rank = 8 - row
        return f"{file}{rank}"


def parse_pgn_to_fens(pgn_path: Path) -> List[str]:
    """
    Parse a PGN file and return all FEN positions (board part only).
    """
    with open(pgn_path, 'r') as f:
        pgn_content = f.read()
    
    game = chess.pgn.read_game(io.StringIO(pgn_content))
    if game is None:
        raise ValueError(f"Could not parse PGN: {pgn_path}")
    
    fens = []
    board = game.board()
    
    # Starting position
    fens.append(board.board_fen())
    
    # Play through all moves
    for move in game.mainline_moves():
        board.push(move)
        fens.append(board.board_fen())
    
    return fens


def get_sorted_frames(images_dir: Path) -> List[Path]:
    """Get all frame images sorted by frame number."""
    frames = list(images_dir.glob("frame_*.jpg"))
    frames.sort(key=lambda p: int(re.search(r'frame_(\d+)', p.stem).group(1)))
    return frames


def sample_frames_for_positions(frames: List[Path], num_positions: int, 
                                samples_per_position: int = 1) -> List[Tuple[Path, int]]:
    """
    Sample frames evenly across positions.
    """
    if len(frames) < num_positions:
        return [(f, i) for i, f in enumerate(frames) if i < num_positions]
    
    samples = []
    frames_per_position = len(frames) / num_positions
    
    for pos_idx in range(num_positions):
        start_idx = int(pos_idx * frames_per_position)
        end_idx = int((pos_idx + 1) * frames_per_position)
        mid_idx = (start_idx + end_idx) // 2
        
        for offset in range(samples_per_position):
            sample_idx = mid_idx + offset
            if sample_idx < len(frames):
                samples.append((frames[sample_idx], pos_idx))
    
    return samples


def process_pgn_game(game_dir: Path, 
                     output_dir: Path,
                     detector: BoardDetector,
                     extractor: BlockSquareExtractor,
                     samples_per_position: int = 1) -> Dict:
    """
    Process a single PGN game and add to the block dataset.
    """
    game_name = game_dir.name
    pgn_path = game_dir / f"{game_name}.pgn"
    images_dir = game_dir / "images"
    
    if not pgn_path.exists():
        print(f"  PGN not found: {pgn_path}")
        return {'success': 0, 'failed': 0, 'total_squares': 0}
    
    if not images_dir.exists():
        print(f"  Images not found: {images_dir}")
        return {'success': 0, 'failed': 0, 'total_squares': 0}
    
    # Parse PGN
    try:
        fens = parse_pgn_to_fens(pgn_path)
    except Exception as e:
        print(f"  Error parsing PGN: {e}")
        return {'success': 0, 'failed': 0, 'total_squares': 0}
    
    print(f"  Found {len(fens)} positions in PGN")
    
    # Get frames and sample
    frames = get_sorted_frames(images_dir)
    print(f"  Found {len(frames)} frames")
    
    samples = sample_frames_for_positions(frames, len(fens), samples_per_position)
    print(f"  Sampling {len(samples)} frame-position pairs")
    
    stats = {'success': 0, 'failed': 0, 'total_squares': 0}
    
    for frame_path, pos_idx in tqdm(samples, desc=f"  {game_name}"):
        fen = fens[pos_idx]
        frame_num = int(re.search(r'frame_(\d+)', frame_path.stem).group(1))
        
        # Load image
        image = cv2.imread(str(frame_path))
        if image is None:
            stats['failed'] += 1
            continue
        
        # Detect and warp board
        warped = detector.detect_board(image, debug=False)
        if warped is None:
            stats['failed'] += 1
            continue
        
        # Extract blocks
        blocks = extractor.extract_blocks(warped)
        
        # Get labels from FEN
        try:
            labels = FENParser.fen_to_labels(fen)
        except ValueError:
            stats['failed'] += 1
            continue
        
        # Save each block
        for square_idx, (block, label) in enumerate(zip(blocks, labels)):
            position = extractor.get_square_position(square_idx)
            filename = f"{game_name}_frame_{frame_num:06d}_{position}.jpg"
            output_path = output_dir / label / filename
            cv2.imwrite(str(output_path), block)
            stats['total_squares'] += 1
        
        stats['success'] += 1
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process PGN games to block dataset')
    parser.add_argument('--pgn-root', type=str,
                       default='/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon/data/PGN',
                       help='Root directory with PGN game folders')
    parser.add_argument('--output-root', type=str,
                       default='/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon/preprocessed_data_blocks',
                       help='Output directory for block dataset')
    parser.add_argument('--border-color', type=str, default='black',
                       choices=['black', 'white', 'gray'],
                       help='Color for edge padding')
    parser.add_argument('--samples-per-position', type=int, default=1,
                       help='Number of frames to sample per chess position')
    parser.add_argument('--games', type=str, nargs='+', default=None,
                       help='Specific games to process (e.g., game8 game9)')
    
    args = parser.parse_args()
    
    pgn_root = Path(args.pgn_root)
    output_root = Path(args.output_root)
    train_dir = output_root / 'train'
    
    print("="*70)
    print("PROCESSING PGN GAMES TO BLOCK DATASET")
    print("="*70)
    print(f"PGN Root: {pgn_root}")
    print(f"Output: {output_root}")
    print(f"Border color: {args.border_color}")
    print()
    
    # Initialize
    detector = BoardDetector(board_size=512)
    extractor = BlockSquareExtractor(board_size=512, border_color=args.border_color)
    
    print(f"Block size: {extractor.block_size}x{extractor.block_size} (3×3 squares)")
    print()
    
    # Ensure output directories exist
    for class_name in FENParser.get_piece_classes():
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    # Find all game directories
    game_dirs = []
    for subdir in ['c06', 'c17']:
        subdir_path = pgn_root / subdir
        if subdir_path.exists():
            for game_dir in subdir_path.iterdir():
                if game_dir.is_dir() and game_dir.name.startswith('game'):
                    if args.games is None or game_dir.name in args.games:
                        game_dirs.append(game_dir)
    
    game_dirs.sort(key=lambda p: int(re.search(r'game(\d+)', p.name).group(1)))
    
    print(f"Found {len(game_dirs)} games to process:")
    for gd in game_dirs:
        print(f"  - {gd.name}")
    print()
    
    # Process each game
    total_stats = {'success': 0, 'failed': 0, 'total_squares': 0}
    
    for game_dir in game_dirs:
        print(f"\nProcessing {game_dir.name}...")
        stats = process_pgn_game(
            game_dir, train_dir, detector, extractor,
            samples_per_position=args.samples_per_position
        )
        
        print(f"  Success: {stats['success']}, Failed: {stats['failed']}, Blocks: {stats['total_squares']}")
        
        total_stats['success'] += stats['success']
        total_stats['failed'] += stats['failed']
        total_stats['total_squares'] += stats['total_squares']
    
    # Print final statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    print(f"Total frames processed: {total_stats['success']}")
    print(f"Total frames failed: {total_stats['failed']}")
    print(f"Total new blocks added: {total_stats['total_squares']}")
    print(f"\nOutput directory: {output_root}")
    
    # Print updated class distribution
    print("\n" + "="*70)
    print("UPDATED CLASS DISTRIBUTION")
    print("="*70)
    for class_dir in sorted(train_dir.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.jpg')))
            print(f"  {class_dir.name:20s}: {count:6d} images")
    
    print("\n✅ PGN games added to block dataset!")
    print("\nNext steps:")
    print("  1. Re-run split_dataset.py to include new games in splits")
    print("  2. Train models on the expanded dataset")


if __name__ == "__main__":
    main()
