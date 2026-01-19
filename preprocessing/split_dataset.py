"""
Dataset Splitting Script

This script splits the preprocessed square images into train/val/test sets.

CRITICAL: We split by GAME, not by individual frames, to prevent data leakage.
Consecutive frames from the same game are highly correlated and would lead to
artificially inflated validation/test performance.

Strategy:
- Split by game to ensure no leakage
- Maintain class distribution across splits
- Create symlinks or copy files to new directory structure
"""

import shutil
import random
from pathlib import Path
from collections import defaultdict
import json
from typing import Dict, List, Tuple


class DatasetSplitter:
    """
    Splits preprocessed data into train/val/test sets by game.
    """
    
    def __init__(self, 
                 preprocessed_root: str,
                 output_root: str,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 seed: int = 42,
                 copy_files: bool = True):
        """
        Initialize the dataset splitter.
        
        Args:
            preprocessed_root: Root directory with preprocessed_data/train/
            output_root: Root directory for split dataset
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for test
            seed: Random seed for reproducibility
            copy_files: If True, copy files. If False, create symlinks.
        """
        self.preprocessed_root = Path(preprocessed_root)
        self.output_root = Path(output_root)
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.train_ratio = 1.0 - val_ratio - test_ratio
        self.seed = seed
        self.copy_files = copy_files
        
        random.seed(seed)
        
        # Source directory
        self.source_dir = self.preprocessed_root / 'train'
        
        # Output directories
        self.train_dir = self.output_root / 'train'
        self.val_dir = self.output_root / 'val'
        self.test_dir = self.output_root / 'test'
        
    def _get_games_from_filename(self, filename: str) -> str:
        """
        Extract game name from filename.
        Format: game2_frame_000200_a8.jpg -> game2
        """
        return filename.split('_')[0]
    
    def _get_all_games(self) -> List[str]:
        """
        Get list of all unique games in the dataset.
        """
        games = set()
        
        # Check all class folders
        for class_dir in self.source_dir.iterdir():
            if class_dir.is_dir():
                for img_file in class_dir.glob('*.jpg'):
                    game = self._get_games_from_filename(img_file.name)
                    games.add(game)
        
        return sorted(list(games))
    
    def _assign_games_to_splits(self, games: List[str]) -> Dict[str, str]:
        """
        Assign each game to train/val/test split.
        
        Strategy:
        1. Shuffle games randomly
        2. Assign based on ratios
        
        Args:
            games: List of game names
            
        Returns:
            Dictionary mapping game_name -> split ('train', 'val', or 'test')
        """
        # Shuffle games
        shuffled_games = games.copy()
        random.shuffle(shuffled_games)
        
        n_games = len(shuffled_games)
        n_val = max(1, int(n_games * self.val_ratio))
        n_test = max(1, int(n_games * self.test_ratio))
        n_train = n_games - n_val - n_test
        
        print(f"\nGame Split:")
        print(f"  Train: {n_train} games")
        print(f"  Val:   {n_val} games")
        print(f"  Test:  {n_test} games")
        
        # Assign splits
        game_splits = {}
        idx = 0
        
        # Train games
        for i in range(n_train):
            game_splits[shuffled_games[idx]] = 'train'
            idx += 1
        
        # Val games
        for i in range(n_val):
            game_splits[shuffled_games[idx]] = 'val'
            idx += 1
        
        # Test games
        for i in range(n_test):
            game_splits[shuffled_games[idx]] = 'test'
            idx += 1
        
        # Print assignment
        print("\nGame Assignments:")
        for split in ['train', 'val', 'test']:
            games_in_split = [g for g, s in game_splits.items() if s == split]
            print(f"  {split.upper():5s}: {', '.join(games_in_split)}")
        
        return game_splits
    
    def _create_output_structure(self):
        """Create output directory structure."""
        print("\nCreating output directory structure...")
        
        # Get all class names
        classes = [d.name for d in self.source_dir.iterdir() if d.is_dir()]
        
        # Create directories for each split and class
        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
            split_dir.mkdir(parents=True, exist_ok=True)
            for class_name in classes:
                (split_dir / class_name).mkdir(exist_ok=True)
        
        print(f"  Created {len(classes)} class folders in each split")
    
    def _split_files(self, game_splits: Dict[str, str]) -> Dict[str, Dict[str, int]]:
        """
        Split files according to game assignments.
        
        Args:
            game_splits: Dictionary mapping game_name -> split
            
        Returns:
            Statistics dictionary
        """
        print("\nSplitting files...")
        
        # Statistics
        stats = {
            'train': defaultdict(int),
            'val': defaultdict(int),
            'test': defaultdict(int)
        }
        
        # Process each class
        for class_dir in self.source_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            print(f"  Processing {class_name}...")
            
            # Process each image
            for img_file in class_dir.glob('*.jpg'):
                # Determine which split this image belongs to
                game = self._get_games_from_filename(img_file.name)
                split = game_splits[game]
                
                # Destination path
                if split == 'train':
                    dest = self.train_dir / class_name / img_file.name
                elif split == 'val':
                    dest = self.val_dir / class_name / img_file.name
                else:  # test
                    dest = self.test_dir / class_name / img_file.name
                
                # Copy or symlink
                if self.copy_files:
                    shutil.copy2(img_file, dest)
                else:
                    dest.symlink_to(img_file.absolute())
                
                # Update statistics
                stats[split][class_name] += 1
        
        return stats
    
    def _print_statistics(self, stats: Dict[str, Dict[str, int]]):
        """Print split statistics."""
        print("\n" + "="*70)
        print("SPLIT STATISTICS")
        print("="*70)
        
        # Get all classes
        all_classes = set()
        for split_stats in stats.values():
            all_classes.update(split_stats.keys())
        all_classes = sorted(all_classes)
        
        # Print header
        print(f"\n{'Class':<20s} {'Train':>10s} {'Val':>10s} {'Test':>10s} {'Total':>10s}")
        print("-" * 70)
        
        # Print per-class stats
        totals = {'train': 0, 'val': 0, 'test': 0}
        for class_name in all_classes:
            train_count = stats['train'].get(class_name, 0)
            val_count = stats['val'].get(class_name, 0)
            test_count = stats['test'].get(class_name, 0)
            total = train_count + val_count + test_count
            
            totals['train'] += train_count
            totals['val'] += val_count
            totals['test'] += test_count
            
            print(f"{class_name:<20s} {train_count:>10d} {val_count:>10d} {test_count:>10d} {total:>10d}")
        
        # Print totals
        print("-" * 70)
        total_all = totals['train'] + totals['val'] + totals['test']
        print(f"{'TOTAL':<20s} {totals['train']:>10d} {totals['val']:>10d} {totals['test']:>10d} {total_all:>10d}")
        
        # Print percentages
        print("\nSplit Percentages:")
        print(f"  Train: {100 * totals['train'] / total_all:.1f}%")
        print(f"  Val:   {100 * totals['val'] / total_all:.1f}%")
        print(f"  Test:  {100 * totals['test'] / total_all:.1f}%")
    
    def _save_split_info(self, game_splits: Dict[str, str], stats: Dict[str, Dict[str, int]]):
        """Save split information to JSON for reproducibility."""
        info = {
            'seed': self.seed,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'game_assignments': game_splits,
            'statistics': {
                split: dict(split_stats) for split, split_stats in stats.items()
            }
        }
        
        info_file = self.output_root / 'split_info.json'
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\nSplit info saved to: {info_file}")
    
    def split(self):
        """
        Main method to split the dataset.
        """
        print("="*70)
        print("DATASET SPLITTING")
        print("="*70)
        print(f"Source: {self.source_dir}")
        print(f"Output: {self.output_root}")
        print(f"Ratios: Train={self.train_ratio:.0%}, Val={self.val_ratio:.0%}, Test={self.test_ratio:.0%}")
        print(f"Seed: {self.seed}")
        print(f"Mode: {'Copy' if self.copy_files else 'Symlink'}")
        
        # Get all games
        games = self._get_all_games()
        print(f"\nFound {len(games)} games: {', '.join(games)}")
        
        # Assign games to splits
        game_splits = self._assign_games_to_splits(games)
        
        # Create output structure
        self._create_output_structure()
        
        # Split files
        stats = self._split_files(game_splits)
        
        # Print statistics
        self._print_statistics(stats)
        
        # Save split info
        self._save_split_info(game_splits, stats)
        
        print("\n" + "="*70)
        print("SPLITTING COMPLETE!")
        print("="*70)
        print(f"\nDataset is ready at: {self.output_root}")
        print("\nDirectory structure:")
        print(f"  {self.output_root}/")
        print(f"    ├── train/")
        print(f"    ├── val/")
        print(f"    ├── test/")
        print(f"    └── split_info.json")


def main():
    """
    Main entry point for dataset splitting.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Split dataset by game')
    parser.add_argument('--preprocessed-root', type=str,
                       default='/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon/preprocessed_data',
                       help='Root directory with preprocessed_data/train/')
    parser.add_argument('--output-root', type=str,
                       default='/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon/dataset',
                       help='Output directory for split dataset')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--symlink', action='store_true',
                       help='Use symlinks instead of copying files')
    
    args = parser.parse_args()
    
    # Create splitter
    splitter = DatasetSplitter(
        preprocessed_root=args.preprocessed_root,
        output_root=args.output_root,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        copy_files=not args.symlink
    )
    
    # Split dataset
    splitter.split()
    
    print("\n✅ Dataset split successfully!")
    print("\nNext steps:")
    print("  1. Verify the split by checking a few samples from each set")
    print("  2. Start training your classifier on the train set")
    print("  3. Use val set for hyperparameter tuning")
    print("  4. ONLY use test set for final evaluation")


if __name__ == "__main__":
    main()

