"""
Main Preprocessing Script

This script processes all labeled game data:
1. Loads images and their corresponding FEN labels from CSV files
2. Detects and warps the chessboard
3. Extracts 64 squares from each board
4. Labels each square based on FEN notation
5. Saves the processed data in an organized structure

Output structure:
    preprocessed_data/
        train/
            white_pawn/
                game2_frame_000200_a2.jpg
                game2_frame_000200_b2.jpg
                ...
            black_knight/
                ...
            empty/
                ...
        metadata/
            game2_metadata.csv (frame, fen, success)
            game4_metadata.csv
            ...
"""

import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import shutil

try:
    from .board_detector import BoardDetector
    from .square_extractor import SquareExtractor, FENParser
except ImportError:
    from board_detector import BoardDetector
    from square_extractor import SquareExtractor, FENParser


class ChessDataPreprocessor:
    """
    Main preprocessor for chessboard data.
    """
    
    def __init__(self, 
                 data_root: str,
                 output_root: str,
                 board_size: int = 512,
                 skip_existing: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            data_root: Root directory containing per_frame data
            output_root: Root directory for preprocessed output
            board_size: Size for warped boards
            skip_existing: If True, skip already processed frames
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.board_size = board_size
        self.skip_existing = skip_existing
        
        # Initialize detector and extractor
        self.detector = BoardDetector(board_size=board_size)
        self.extractor = SquareExtractor(board_size=board_size)
        
        # Create output directories
        self._setup_output_dirs()
        
    def _setup_output_dirs(self):
        """Create output directory structure."""
        # Main directories
        self.train_dir = self.output_root / 'train'
        self.metadata_dir = self.output_root / 'metadata'
        self.warped_boards_dir = self.output_root / 'warped_boards'
        self.failed_dir = self.output_root / 'failed_detections'
        
        # Create directories
        for dir_path in [self.train_dir, self.metadata_dir, 
                        self.warped_boards_dir, self.failed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create class directories
        for class_name in FENParser.get_piece_classes():
            (self.train_dir / class_name).mkdir(exist_ok=True)
    
    def process_game(self, game_dir: Path) -> Dict:
        """
        Process a single game directory.
        
        Args:
            game_dir: Path to game directory (e.g., game2_per_frame)
            
        Returns:
            Dictionary with processing statistics
        """
        game_name = game_dir.name.replace('_per_frame', '')
        print(f"\n{'='*60}")
        print(f"Processing {game_name}")
        print(f"{'='*60}")
        
        # Find CSV file
        csv_file = game_dir / f"{game_name}.csv"
        if not csv_file.exists():
            print(f"CSV file not found: {csv_file}")
            return {'success': 0, 'failed': 0, 'skipped': 0}
        
        # Load CSV
        df = pd.read_csv(csv_file)
        print(f"Found {len(df)} labeled frames")
        
        # Statistics
        stats = {
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'total_squares': 0
        }
        
        # Metadata for this game
        metadata = []
        
        # Process each frame
        images_dir = game_dir / 'tagged_images'
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{game_name}"):
            frame_num = row['from_frame']
            fen = row['fen']
            
            # Find image file
            image_path = images_dir / f"frame_{frame_num:06d}.jpg"
            if not image_path.exists():
                print(f"Image not found: {image_path}")
                stats['failed'] += 1
                metadata.append({
                    'frame': frame_num,
                    'fen': fen,
                    'status': 'image_not_found'
                })
                continue
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Failed to load image: {image_path}")
                stats['failed'] += 1
                metadata.append({
                    'frame': frame_num,
                    'fen': fen,
                    'status': 'load_failed'
                })
                continue
            
            # Detect and warp board
            warped = self.detector.detect_board(image, debug=False)
            
            if warped is None:
                print(f"Failed to detect board in frame {frame_num}")
                stats['failed'] += 1
                
                # Save failed image for inspection
                failed_path = self.failed_dir / f"{game_name}_frame_{frame_num:06d}.jpg"
                cv2.imwrite(str(failed_path), image)
                
                metadata.append({
                    'frame': frame_num,
                    'fen': fen,
                    'status': 'detection_failed'
                })
                continue
            
            # Save warped board for inspection
            warped_path = self.warped_boards_dir / f"{game_name}_frame_{frame_num:06d}.jpg"
            cv2.imwrite(str(warped_path), warped)
            
            # Extract squares
            squares = self.extractor.extract_squares(warped)
            
            # Get labels from FEN
            try:
                labels = FENParser.fen_to_labels(fen)
            except ValueError as e:
                print(f"Invalid FEN in frame {frame_num}: {e}")
                stats['failed'] += 1
                metadata.append({
                    'frame': frame_num,
                    'fen': fen,
                    'status': 'invalid_fen'
                })
                continue
            
            # Save each square
            for square_idx, (square, label) in enumerate(zip(squares, labels)):
                position = self.extractor.get_square_position(square_idx)
                
                # Create filename: game_frame_position.jpg
                filename = f"{game_name}_frame_{frame_num:06d}_{position}.jpg"
                output_path = self.train_dir / label / filename
                
                # Save square image
                cv2.imwrite(str(output_path), square)
                stats['total_squares'] += 1
            
            stats['success'] += 1
            metadata.append({
                'frame': frame_num,
                'fen': fen,
                'status': 'success'
            })
        
        # Save metadata
        metadata_df = pd.DataFrame(metadata)
        metadata_path = self.metadata_dir / f"{game_name}_metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        
        print(f"\n{game_name} Statistics:")
        print(f"  Success: {stats['success']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Total squares extracted: {stats['total_squares']}")
        
        return stats
    
    def process_all_games(self):
        """
        Process all games in the per_frame directory.
        """
        per_frame_dir = self.data_root / 'per_frame'
        
        if not per_frame_dir.exists():
            print(f"per_frame directory not found: {per_frame_dir}")
            return
        
        # Find all game directories
        game_dirs = [d for d in per_frame_dir.iterdir() 
                    if d.is_dir() and 'game' in d.name and '_per_frame' in d.name]
        
        print(f"Found {len(game_dirs)} games to process")
        
        # Overall statistics
        total_stats = {
            'success': 0,
            'failed': 0,
            'total_squares': 0
        }
        
        # Process each game
        for game_dir in sorted(game_dirs):
            stats = self.process_game(game_dir)
            total_stats['success'] += stats['success']
            total_stats['failed'] += stats['failed']
            total_stats['total_squares'] += stats['total_squares']
        
        # Print final statistics
        print(f"\n{'='*60}")
        print("FINAL STATISTICS")
        print(f"{'='*60}")
        print(f"Total frames processed successfully: {total_stats['success']}")
        print(f"Total frames failed: {total_stats['failed']}")
        print(f"Total square images extracted: {total_stats['total_squares']}")
        
        # Print class distribution
        self._print_class_distribution()
    
    def _print_class_distribution(self):
        """Print the distribution of square images across classes."""
        print(f"\n{'='*60}")
        print("CLASS DISTRIBUTION")
        print(f"{'='*60}")
        
        class_counts = {}
        for class_dir in sorted(self.train_dir.iterdir()):
            if class_dir.is_dir():
                count = len(list(class_dir.glob('*.jpg')))
                class_counts[class_dir.name] = count
        
        # Sort by count
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        
        for class_name, count in sorted_classes:
            print(f"  {class_name:20s}: {count:6d} images")
        
        total = sum(class_counts.values())
        print(f"\n  {'TOTAL':20s}: {total:6d} images")


def main():
    """
    Main entry point for preprocessing.
    """
    # Paths
    data_root = "/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon/data"
    output_root = "/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon/preprocessed_data"
    
    # Create preprocessor
    preprocessor = ChessDataPreprocessor(
        data_root=data_root,
        output_root=output_root,
        board_size=512,
        skip_existing=True
    )
    
    # Process all games
    preprocessor.process_all_games()
    
    print("\nPreprocessing complete!")
    print(f"Output saved to: {output_root}")


if __name__ == "__main__":
    main()

