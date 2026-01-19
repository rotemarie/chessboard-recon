"""
Process PGN Games for Padded Dataset

This script processes games from the PGN folder structure and adds them
to the existing padded dataset. It:
1. Parses PGN files to get FEN positions for each move
2. Maps frames to positions (samples frames evenly across the game)
3. Creates padded square images and adds them to the dataset

Usage:
    python process_pgn_games.py --output-root ../preprocessed_data_padded
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
    
    def extract_squares_with_padding(self, warped_board: np.ndarray) -> list:
        """
        Extract all 64 squares with padding from a warped board.
        """
        padded_board = cv2.copyMakeBorder(
            warped_board,
            self.padding_pixels,
            self.padding_pixels,
            self.padding_pixels,
            self.padding_pixels,
            cv2.BORDER_CONSTANT,
            value=self.border_color
        )
        
        squares = []
        for row in range(8):
            for col in range(8):
                y1 = row * self.square_size
                y2 = y1 + self.output_size
                x1 = col * self.square_size
                x2 = x1 + self.output_size
                square = padded_board[y1:y2, x1:x2]
                squares.append(square)
        
        return squares
    
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
    
    Args:
        pgn_path: Path to PGN file
        
    Returns:
        List of FEN strings (just the piece placement part)
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
    
    Args:
        frames: List of frame paths
        num_positions: Number of FEN positions
        samples_per_position: How many frames to sample per position
        
    Returns:
        List of (frame_path, position_index) tuples
    """
    if len(frames) < num_positions:
        # Fewer frames than positions - use each frame once
        return [(f, i) for i, f in enumerate(frames) if i < num_positions]
    
    samples = []
    frames_per_position = len(frames) / num_positions
    
    for pos_idx in range(num_positions):
        # Sample from the middle of each position's frame range
        start_idx = int(pos_idx * frames_per_position)
        end_idx = int((pos_idx + 1) * frames_per_position)
        mid_idx = (start_idx + end_idx) // 2
        
        # Sample around the middle
        for offset in range(samples_per_position):
            sample_idx = mid_idx + offset
            if sample_idx < len(frames):
                samples.append((frames[sample_idx], pos_idx))
    
    return samples


def process_pgn_game(game_dir: Path, 
                     output_dir: Path,
                     detector: BoardDetector,
                     extractor: PaddedSquareExtractor,
                     samples_per_position: int = 1) -> Dict:
    """
    Process a single PGN game and add to the padded dataset.
    
    Args:
        game_dir: Directory containing game.pgn and images/
        output_dir: Output directory (train folder)
        detector: Board detector
        extractor: Padded square extractor
        samples_per_position: How many frames to sample per chess position
        
    Returns:
        Statistics dictionary
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
    
    # Parse PGN to get all positions
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
        
        # Extract padded squares
        squares = extractor.extract_squares_with_padding(warped)
        
        # Get labels from FEN
        try:
            labels = FENParser.fen_to_labels(fen)
        except ValueError:
            stats['failed'] += 1
            continue
        
        # Save each square
        for square_idx, (square, label) in enumerate(zip(squares, labels)):
            position = extractor.get_square_position(square_idx)
            filename = f"{game_name}_frame_{frame_num:06d}_{position}.jpg"
            output_path = output_dir / label / filename
            cv2.imwrite(str(output_path), square)
            stats['total_squares'] += 1
        
        stats['success'] += 1
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process PGN games to padded dataset')
    parser.add_argument('--pgn-root', type=str,
                       default='/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon/data/PGN',
                       help='Root directory with PGN game folders')
    parser.add_argument('--output-root', type=str,
                       default='/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon/preprocessed_data_padded',
                       help='Output directory for padded dataset')
    parser.add_argument('--padding', type=float, default=0.30,
                       help='Padding percentage (0.30 = 30%)')
    parser.add_argument('--samples-per-position', type=int, default=1,
                       help='Number of frames to sample per chess position')
    parser.add_argument('--games', type=str, nargs='+', default=None,
                       help='Specific games to process (e.g., game8 game9)')
    
    args = parser.parse_args()
    
    pgn_root = Path(args.pgn_root)
    output_root = Path(args.output_root)
    train_dir = output_root / 'train'
    
    print("="*70)
    print("PROCESSING PGN GAMES TO PADDED DATASET")
    print("="*70)
    print(f"PGN Root: {pgn_root}")
    print(f"Output: {output_root}")
    print(f"Padding: {args.padding*100:.0f}%")
    print()
    
    # Initialize
    detector = BoardDetector(board_size=512)
    extractor = PaddedSquareExtractor(board_size=512, 
                                      padding_percent=args.padding,
                                      border_color='black')
    
    print(f"Output size: {extractor.output_size}x{extractor.output_size}")
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
        
        print(f"  Success: {stats['success']}, Failed: {stats['failed']}, Squares: {stats['total_squares']}")
        
        total_stats['success'] += stats['success']
        total_stats['failed'] += stats['failed']
        total_stats['total_squares'] += stats['total_squares']
    
    # Print final statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    print(f"Total frames processed: {total_stats['success']}")
    print(f"Total frames failed: {total_stats['failed']}")
    print(f"Total new squares added: {total_stats['total_squares']}")
    print(f"\nOutput directory: {output_root}")
    
    # Print updated class distribution
    print("\n" + "="*70)
    print("UPDATED CLASS DISTRIBUTION")
    print("="*70)
    for class_dir in sorted(train_dir.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.jpg')))
            print(f"  {class_dir.name:20s}: {count:6d} images")
    
    print("\n✅ PGN games added to padded dataset!")
    print("\nNext steps:")
    print("  1. Re-run split_dataset.py to include new games in splits")
    print("  2. Train models on the expanded dataset")


if __name__ == "__main__":
    main()
