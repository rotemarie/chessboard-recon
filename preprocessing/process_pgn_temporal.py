"""
Process PGN Games using Temporal Analysis

Batch processes all PGN games using the temporal tagger.
Generates properly aligned frame-FEN mappings for each game.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import argparse
from typing import Dict

from temporal_tagger import StabilityDetector, FENAligner, parse_pgn_to_fens
from board_detector import BoardDetector
from square_extractor import SquareExtractor, FENParser


def find_video_file(game_dir: Path) -> Path:
    """
    Find video file in game directory (handles different extensions).
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    # Try images/ folder first (might contain video)
    images_dir = game_dir / 'images'
    if images_dir.exists():
        for ext in video_extensions:
            video_files = list(images_dir.glob(f'*{ext}'))
            if video_files:
                return video_files[0]
    
    # Try game directory
    for ext in video_extensions:
        video_files = list(game_dir.glob(f'*{ext}'))
        if video_files:
            return video_files[0]
    
    # No video - will need to create from frames
    return None


def create_video_from_frames(frames_dir: Path, output_path: Path, fps: int = 30) -> bool:
    """
    Create video from image sequence if no video file exists.
    """
    print(f"  Creating video from frames in {frames_dir}")
    
    # Get all frames
    frame_files = sorted(frames_dir.glob('frame_*.jpg'))
    if not frame_files:
        print(f"  No frames found in {frames_dir}")
        return False
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        return False
    
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Write all frames
    for frame_file in tqdm(frame_files, desc="  Creating video"):
        frame = cv2.imread(str(frame_file))
        if frame is not None:
            out.write(frame)
    
    out.release()
    print(f"  Video created: {output_path}")
    return True


def process_single_game(game_dir: Path,
                       output_dir: Path,
                       aligner: FENAligner,
                       detector: StabilityDetector,
                       args: argparse.Namespace) -> Dict:
    """
    Process a single PGN game.
    
    Returns:
        Statistics dictionary
    """
    game_name = game_dir.name
    print(f"\n{'='*70}")
    print(f"Processing {game_name}")
    print('='*70)
    
    # Find PGN file
    pgn_file = game_dir / f"{game_name}.pgn"
    if not pgn_file.exists():
        print(f"  PGN not found: {pgn_file}")
        return {'success': False, 'reason': 'no_pgn'}
    
    # Parse PGN
    try:
        pgn_positions = parse_pgn_to_fens(pgn_file)
        print(f"  Found {len(pgn_positions)} positions in PGN")
    except Exception as e:
        print(f"  Error parsing PGN: {e}")
        return {'success': False, 'reason': 'pgn_parse_error'}
    
    # Find or create video
    video_file = find_video_file(game_dir)
    
    if video_file is None:
        # Try to create from frames
        frames_dir = game_dir / 'images'
        if not frames_dir.exists():
            print(f"  No video or frames found")
            return {'success': False, 'reason': 'no_video_or_frames'}
        
        video_file = output_dir / 'temp_videos' / f"{game_name}.mp4"
        video_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not create_video_from_frames(frames_dir, video_file):
            return {'success': False, 'reason': 'video_creation_failed'}
    
    print(f"  Video: {video_file}")
    
    # Detect stable segments
    try:
        segments, frames = detector.detect_stable_segments(str(video_file))
        print(f"  Found {len(segments)} stable segments")
    except Exception as e:
        print(f"  Error detecting segments: {e}")
        return {'success': False, 'reason': 'detection_error', 'error': str(e)}
    
    # Align to PGN
    try:
        alignments = aligner.align_segments_to_pgn(
            segments, frames, pgn_positions,
            min_similarity=args.min_similarity
        )
        
        if not alignments:
            print(f"  No alignments found")
            return {'success': False, 'reason': 'no_alignments'}
        
        print(f"  Successfully aligned {len(alignments)}/{len(pgn_positions)} positions")
        avg_similarity = np.mean([s for _, _, s in alignments])
        print(f"  Average similarity: {avg_similarity:.2%}")
    except Exception as e:
        print(f"  Error aligning: {e}")
        return {'success': False, 'reason': 'alignment_error', 'error': str(e)}
    
    # Save results
    csv_file = output_dir / 'tagged_csvs' / f"{game_name}_tagged.csv"
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_file, 'w') as f:
        f.write("frame_idx,fen,similarity\n")
        for frame_idx, fen, similarity in alignments:
            f.write(f"{frame_idx},{fen},{similarity:.4f}\n")
    
    print(f"  Saved: {csv_file}")
    
    # Save representative frames
    if args.save_frames:
        frames_out_dir = output_dir / 'tagged_frames' / game_name
        frames_out_dir.mkdir(parents=True, exist_ok=True)
        
        for frame_idx, fen, _ in alignments[:min(10, len(alignments))]:  # Save first 10
            frame_file = frames_out_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_file), frames[frame_idx])
    
    return {
        'success': True,
        'game': game_name,
        'total_pgn_positions': len(pgn_positions),
        'stable_segments': len(segments),
        'alignments': len(alignments),
        'avg_similarity': avg_similarity,
        'coverage': len(alignments) / len(pgn_positions)
    }


def main():
    parser = argparse.ArgumentParser(description='Process PGN games with temporal analysis')
    parser.add_argument('--pgn-root', type=str,
                       default='/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon/data/PGN',
                       help='Root directory with PGN games')
    parser.add_argument('--model', type=str,
                       default='../model/resnet18_ft.pth',
                       help='Path to trained model')
    parser.add_argument('--classes', type=str,
                       default='../model/classes.txt',
                       help='Path to classes.txt')
    parser.add_argument('--output', type=str,
                       default='../pgn_temporal_output',
                       help='Output directory')
    parser.add_argument('--games', type=str, nargs='+', default=None,
                       help='Specific games to process (e.g., game8 game9)')
    parser.add_argument('--flow-threshold', type=float, default=25,
                       help='Flow percentile threshold (default: 25)')
    parser.add_argument('--min-stable-frames', type=int, default=15,
                       help='Minimum stable frames (default: 15)')
    parser.add_argument('--min-similarity', type=float, default=0.80,
                       help='Minimum FEN similarity for alignment (default: 0.80)')
    parser.add_argument('--save-frames', action='store_true',
                       help='Save representative frames for each game')
    
    args = parser.parse_args()
    
    pgn_root = Path(args.pgn_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("TEMPORAL PGN PROCESSING")
    print("="*70)
    print(f"PGN Root: {pgn_root}")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")
    print(f"Flow threshold: P{args.flow_threshold}")
    print(f"Min stable frames: {args.min_stable_frames}")
    print(f"Min similarity: {args.min_similarity}")
    print()
    
    # Initialize components
    print("Initializing...")
    detector = StabilityDetector(
        flow_threshold_percentile=args.flow_threshold,
        min_stable_frames=args.min_stable_frames
    )
    
    board_detector = BoardDetector(board_size=512)
    square_extractor = SquareExtractor(board_size=512)
    
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    aligner = FENAligner(
        model_path=args.model,
        classes_file=args.classes,
        board_detector=board_detector,
        square_extractor=square_extractor,
        device=device
    )
    
    # Find games
    game_dirs = []
    for subdir in ['c06', 'c17']:
        subdir_path = pgn_root / subdir
        if subdir_path.exists():
            for game_dir in subdir_path.iterdir():
                if game_dir.is_dir() and game_dir.name.startswith('game'):
                    if args.games is None or game_dir.name in args.games:
                        game_dirs.append(game_dir)
    
    game_dirs.sort(key=lambda p: int(p.name.replace('game', '')))
    
    print(f"\nFound {len(game_dirs)} games to process:")
    for gd in game_dirs:
        print(f"  - {gd.name}")
    
    # Process each game
    results = []
    for game_dir in game_dirs:
        result = process_single_game(game_dir, output_dir, aligner, detector, args)
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]
    
    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    
    if successful:
        print("\nPer-Game Statistics:")
        print(f"{'Game':<12} {'PGN Pos':<10} {'Segments':<10} {'Aligned':<10} {'Coverage':<10} {'Avg Sim':<10}")
        print("-" * 70)
        for r in successful:
            print(f"{r['game']:<12} {r['total_pgn_positions']:<10} {r['stable_segments']:<10} "
                  f"{r['alignments']:<10} {r['coverage']:>9.1%} {r['avg_similarity']:>9.1%}")
        
        # Overall stats
        total_pgn = sum(r['total_pgn_positions'] for r in successful)
        total_aligned = sum(r['alignments'] for r in successful)
        avg_coverage = np.mean([r['coverage'] for r in successful])
        avg_similarity = np.mean([r['avg_similarity'] for r in successful])
        
        print("-" * 70)
        print(f"{'TOTAL':<12} {total_pgn:<10} {'':<10} {total_aligned:<10} "
              f"{avg_coverage:>9.1%} {avg_similarity:>9.1%}")
    
    if failed:
        print(f"\nFailed: {len(failed)}")
        for r in failed:
            print(f"  - {r.get('game', 'unknown')}: {r.get('reason', 'unknown error')}")
    
    print(f"\n✅ Processing complete!")
    print(f"Output directory: {output_dir}")
    print(f"  - Tagged CSVs: {output_dir / 'tagged_csvs'}")
    if args.save_frames:
        print(f"  - Sample frames: {output_dir / 'tagged_frames'}")


if __name__ == "__main__":
    main()
