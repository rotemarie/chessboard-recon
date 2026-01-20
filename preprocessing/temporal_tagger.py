"""
Temporal Frame Tagger using Optical Flow

This script uses optical flow to detect stable board positions in chess videos,
then aligns them to PGN positions chronologically. Uses the trained classifier
to validate alignments.

Key advantages over naive sampling:
- Detects actual move boundaries (not just time-based)
- Skips transition frames (pieces in air, hands visible)
- Validates assignments using model predictions
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import chess
import chess.pgn
import io
from tqdm import tqdm

from board_detector import BoardDetector
from square_extractor import SquareExtractor, FENParser


class StabilityDetector:
    """
    Detects stable board positions using optical flow analysis.
    """
    
    def __init__(self, 
                 flow_threshold_percentile: float = 25,
                 min_stable_frames: int = 15,
                 gap_tolerance: int = 10):
        """
        Args:
            flow_threshold_percentile: Percentile below which flow is considered "stable"
            min_stable_frames: Minimum consecutive frames to consider a stable segment
            gap_tolerance: Allow small gaps in stable sequences (for noise)
        """
        self.flow_threshold_percentile = flow_threshold_percentile
        self.min_stable_frames = min_stable_frames
        self.gap_tolerance = gap_tolerance
    
    def compute_optical_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute average optical flow magnitude between two frames.
        
        Args:
            frame1, frame2: BGR images
            
        Returns:
            Average flow magnitude (higher = more motion)
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Calculate magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return np.mean(magnitude)
    
    def detect_stable_segments(self, video_path: str) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
        """
        Detect stable board positions in a video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            - List of (start_frame, end_frame) tuples for stable segments
            - List of all frames
        """
        print(f"Loading video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frames = []
        flow_scores = []
        
        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Video is empty")
        
        frames.append(prev_frame)
        frame_idx = 0
        
        # Process all frames
        pbar = tqdm(desc="Computing optical flow")
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            # Compute flow
            flow_magnitude = self.compute_optical_flow(prev_frame, curr_frame)
            flow_scores.append(flow_magnitude)
            frames.append(curr_frame)
            
            prev_frame = curr_frame
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        print(f"Processed {len(frames)} frames")
        
        # Determine threshold for "stable"
        threshold = np.percentile(flow_scores, self.flow_threshold_percentile)
        print(f"Flow threshold (P{self.flow_threshold_percentile}): {threshold:.2f}")
        
        # Mark stable frames
        is_stable = [score < threshold for score in flow_scores]
        
        # Cluster into segments (allow small gaps)
        segments = self._cluster_stable_frames(is_stable)
        
        print(f"Found {len(segments)} stable segments")
        return segments, frames
    
    def _cluster_stable_frames(self, is_stable: List[bool]) -> List[Tuple[int, int]]:
        """
        Group consecutive stable frames into segments.
        Allows small gaps (gap_tolerance) to handle noise.
        """
        segments = []
        current_segment = []
        gap_count = 0
        
        for idx, stable in enumerate(is_stable):
            if stable:
                current_segment.append(idx)
                gap_count = 0
            else:
                gap_count += 1
                
                # Small gap: keep building segment
                if gap_count <= self.gap_tolerance and current_segment:
                    continue
                
                # Large gap or end: finalize segment
                if len(current_segment) >= self.min_stable_frames:
                    segments.append((current_segment[0], current_segment[-1]))
                
                current_segment = []
                gap_count = 0
        
        # Don't forget last segment
        if len(current_segment) >= self.min_stable_frames:
            segments.append((current_segment[0], current_segment[-1]))
        
        return segments


class FENAligner:
    """
    Aligns detected stable segments to PGN positions using model predictions.
    """
    
    def __init__(self,
                 model_path: str,
                 classes_file: str,
                 board_detector: BoardDetector,
                 square_extractor: SquareExtractor,
                 device: str = 'cpu'):
        """
        Args:
            model_path: Path to trained model checkpoint
            classes_file: Path to classes.txt
            board_detector: BoardDetector instance
            square_extractor: SquareExtractor instance
            device: 'cuda' or 'cpu'
        """
        self.board_detector = board_detector
        self.square_extractor = square_extractor
        self.device = device
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model = self._load_model(model_path, classes_file)
        self.model.eval()
        
        # Load class names
        with open(classes_file, 'r') as f:
            self.class_names = [line.strip() for line in f]
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path: str, classes_file: str) -> nn.Module:
        """Load trained model."""
        # Count classes
        with open(classes_file, 'r') as f:
            num_classes = len(f.readlines())
        
        # Load ResNet18
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model.to(self.device)
    
    def predict_fen(self, frame: np.ndarray) -> Optional[str]:
        """
        Predict FEN for a frame.
        
        Args:
            frame: BGR image
            
        Returns:
            FEN string or None if board detection fails
        """
        # Detect board
        warped = self.board_detector.detect_board(frame, debug=False)
        if warped is None:
            return None
        
        # Extract squares
        squares = self.square_extractor.extract_squares(warped)
        
        # Classify each square
        labels = []
        with torch.no_grad():
            for square in squares:
                # Convert to PIL and preprocess
                square_rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(square_rgb)
                tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                
                # Predict
                output = self.model(tensor)
                pred_idx = output.argmax(dim=1).item()
                labels.append(self.class_names[pred_idx])
        
        # Convert to FEN
        try:
            fen = FENParser.labels_to_fen(labels)
            return fen
        except Exception as e:
            print(f"FEN conversion failed: {e}")
            return None
    
    def compute_fen_similarity(self, fen1: str, fen2: str) -> float:
        """
        Compute similarity between two FENs (% of matching squares).
        
        Args:
            fen1, fen2: FEN strings (board part only)
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            labels1 = FENParser.fen_to_labels(fen1)
            labels2 = FENParser.fen_to_labels(fen2)
            matches = sum(1 for l1, l2 in zip(labels1, labels2) if l1 == l2)
            return matches / 64
        except Exception:
            return 0.0
    
    def align_segments_to_pgn(self,
                              segments: List[Tuple[int, int]],
                              frames: List[np.ndarray],
                              pgn_positions: List[str],
                              min_similarity: float = 0.80) -> List[Tuple[int, str, float]]:
        """
        Align stable segments to PGN positions chronologically.
        
        Args:
            segments: List of (start_frame, end_frame) tuples
            frames: All video frames
            pgn_positions: FEN positions from PGN (in order)
            min_similarity: Minimum similarity to accept alignment
            
        Returns:
            List of (frame_idx, fen, similarity) tuples for accepted alignments
        """
        print(f"\nAligning {len(segments)} segments to {len(pgn_positions)} PGN positions")
        
        alignments = []
        pgn_idx = 0
        
        for seg_start, seg_end in tqdm(segments, desc="Aligning segments"):
            if pgn_idx >= len(pgn_positions):
                break
            
            # Use middle frame of segment
            frame_idx = (seg_start + seg_end) // 2
            frame = frames[frame_idx]
            
            # Predict FEN
            predicted_fen = self.predict_fen(frame)
            if predicted_fen is None:
                continue
            
            # Compare to current PGN position
            expected_fen = pgn_positions[pgn_idx]
            similarity = self.compute_fen_similarity(predicted_fen, expected_fen)
            
            if similarity >= min_similarity:
                # Good match!
                alignments.append((frame_idx, expected_fen, similarity))
                pgn_idx += 1
            else:
                # Try next position (might have missed one)
                if pgn_idx < len(pgn_positions) - 1:
                    next_similarity = self.compute_fen_similarity(
                        predicted_fen, pgn_positions[pgn_idx + 1]
                    )
                    
                    if next_similarity > similarity and next_similarity >= min_similarity:
                        # Better match with next position
                        pgn_idx += 1
                        alignments.append((frame_idx, pgn_positions[pgn_idx], next_similarity))
                        pgn_idx += 1
        
        print(f"Successfully aligned {len(alignments)}/{len(pgn_positions)} positions")
        print(f"Average similarity: {np.mean([s for _, _, s in alignments]):.2%}")
        
        return alignments


def parse_pgn_to_fens(pgn_path: Path) -> List[str]:
    """Parse PGN file and return FEN positions."""
    with open(pgn_path, 'r') as f:
        pgn_content = f.read()
    
    game = chess.pgn.read_game(io.StringIO(pgn_content))
    if game is None:
        raise ValueError(f"Could not parse PGN: {pgn_path}")
    
    fens = []
    board = game.board()
    fens.append(board.board_fen())
    
    for move in game.mainline_moves():
        board.push(move)
        fens.append(board.board_fen())
    
    return fens


def main():
    """
    Main pipeline: detect stable segments, align to PGN, and report statistics.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Tag PGN videos using temporal analysis')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--pgn', type=str, required=True,
                       help='Path to PGN file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--classes', type=str, required=True,
                       help='Path to classes.txt')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file for frame-FEN mappings')
    parser.add_argument('--flow-threshold', type=float, default=25,
                       help='Flow percentile threshold (default: 25)')
    parser.add_argument('--min-stable-frames', type=int, default=15,
                       help='Minimum stable frames (default: 15)')
    parser.add_argument('--min-similarity', type=float, default=0.80,
                       help='Minimum FEN similarity (default: 0.80)')
    
    args = parser.parse_args()
    
    # Initialize components
    detector = StabilityDetector(
        flow_threshold_percentile=args.flow_threshold,
        min_stable_frames=args.min_stable_frames
    )
    
    board_detector = BoardDetector(board_size=512)
    square_extractor = SquareExtractor(board_size=512)
    
    aligner = FENAligner(
        model_path=args.model,
        classes_file=args.classes,
        board_detector=board_detector,
        square_extractor=square_extractor,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Parse PGN
    print(f"\nParsing PGN: {args.pgn}")
    pgn_positions = parse_pgn_to_fens(Path(args.pgn))
    print(f"Found {len(pgn_positions)} positions in PGN")
    
    # Detect stable segments
    print(f"\nAnalyzing video: {args.video}")
    segments, frames = detector.detect_stable_segments(args.video)
    
    # Align to PGN
    alignments = aligner.align_segments_to_pgn(
        segments, frames, pgn_positions,
        min_similarity=args.min_similarity
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("frame_idx,fen,similarity\n")
        for frame_idx, fen, similarity in alignments:
            f.write(f"{frame_idx},{fen},{similarity:.4f}\n")
    
    print(f"\nResults saved to: {output_path}")
    print(f"Tagged {len(alignments)} frames with average similarity {np.mean([s for _, _, s in alignments]):.2%}")


if __name__ == "__main__":
    main()
