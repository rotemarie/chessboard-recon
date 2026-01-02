"""
Square Extraction Module

This module handles:
1. Slicing a warped board image into 64 squares
2. Labeling each square based on FEN notation
3. Saving extracted squares with labels
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import os


class SquareExtractor:
    """
    Extracts individual squares from a warped chessboard image.
    """
    
    def __init__(self, board_size: int = 512):
        """
        Initialize the square extractor.
        
        Args:
            board_size: Size of the warped board (must match BoardDetector)
        """
        self.board_size = board_size
        self.square_size = board_size // 8
        
    def extract_squares(self, warped_board: np.ndarray) -> List[np.ndarray]:
        """
        Extract all 64 squares from a warped board image.
        
        The squares are extracted from rank 8 to rank 1, file a to file h.
        Order: a8, b8, c8, ..., h8, a7, b7, ..., h1
        
        Args:
            warped_board: Top-down view of the board (square image)
            
        Returns:
            List of 64 square images
        """
        squares = []
        
        # Extract squares row by row (from top to bottom = rank 8 to rank 1)
        for row in range(8):
            for col in range(8):
                # Calculate pixel coordinates
                y1 = row * self.square_size
                y2 = (row + 1) * self.square_size
                x1 = col * self.square_size
                x2 = (col + 1) * self.square_size
                
                # Extract square
                square = warped_board[y1:y2, x1:x2]
                squares.append(square)
        
        return squares
    
    def get_square_position(self, square_idx: int) -> str:
        """
        Get chess notation for a square index.
        
        Args:
            square_idx: Index from 0-63 (0=a8, 7=h8, 8=a7, ..., 63=h1)
            
        Returns:
            Chess notation (e.g., 'a8', 'h1')
        """
        row = square_idx // 8  # 0-7 (rank 8 to 1)
        col = square_idx % 8   # 0-7 (file a to h)
        
        file = chr(ord('a') + col)
        rank = 8 - row
        
        return f"{file}{rank}"
    
    def visualize_squares(self, squares: List[np.ndarray], labels: List[str] = None,
                         num_display: int = 16) -> None:
        """
        Visualize extracted squares in a grid.
        
        Args:
            squares: List of square images
            labels: Optional list of labels for each square
            num_display: Number of squares to display
        """
        import matplotlib.pyplot as plt
        
        num_display = min(num_display, len(squares))
        rows = int(np.sqrt(num_display))
        cols = (num_display + rows - 1) // rows
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        axes = axes.flatten() if num_display > 1 else [axes]
        
        for idx in range(num_display):
            ax = axes[idx]
            square = cv2.cvtColor(squares[idx], cv2.COLOR_BGR2RGB)
            ax.imshow(square)
            
            title = self.get_square_position(idx)
            if labels and idx < len(labels):
                title += f"\n{labels[idx]}"
            ax.set_title(title)
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(num_display, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()


class FENParser:
    """
    Parses FEN notation and maps to square labels.
    """
    
    # Mapping from FEN characters to piece names
    PIECE_MAP = {
        'P': 'white_pawn',
        'N': 'white_knight',
        'B': 'white_bishop',
        'R': 'white_rook',
        'Q': 'white_queen',
        'K': 'white_king',
        'p': 'black_pawn',
        'n': 'black_knight',
        'b': 'black_bishop',
        'r': 'black_rook',
        'q': 'black_queen',
        'k': 'black_king',
    }
    
    @staticmethod
    def fen_to_labels(fen: str) -> List[str]:
        """
        Convert FEN notation to a list of 64 labels.
        
        FEN describes the board from rank 8 to rank 1, file a to file h.
        Numbers represent empty squares.
        
        Example FEN: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        
        Args:
            fen: FEN string (only the piece placement part)
            
        Returns:
            List of 64 labels (e.g., ['black_rook', 'empty', ...])
        """
        # Split FEN by ranks
        ranks = fen.split('/')
        
        if len(ranks) != 8:
            raise ValueError(f"Invalid FEN: expected 8 ranks, got {len(ranks)}")
        
        labels = []
        
        for rank in ranks:
            for char in rank:
                if char.isdigit():
                    # Empty squares
                    num_empty = int(char)
                    labels.extend(['empty'] * num_empty)
                else:
                    # Piece
                    piece_name = FENParser.PIECE_MAP.get(char, 'unknown')
                    labels.append(piece_name)
        
        if len(labels) != 64:
            raise ValueError(f"Invalid FEN: expected 64 squares, got {len(labels)}")
        
        return labels
    
    @staticmethod
    def labels_to_fen(labels: List[str]) -> str:
        """
        Convert a list of 64 labels back to FEN notation.
        
        Args:
            labels: List of 64 labels
            
        Returns:
            FEN string
        """
        if len(labels) != 64:
            raise ValueError(f"Expected 64 labels, got {len(labels)}")
        
        # Reverse piece map
        reverse_map = {v: k for k, v in FENParser.PIECE_MAP.items()}
        
        fen_ranks = []
        
        # Process each rank (8 labels at a time)
        for rank_idx in range(8):
            rank_labels = labels[rank_idx * 8:(rank_idx + 1) * 8]
            fen_rank = ""
            empty_count = 0
            
            for label in rank_labels:
                if label == 'empty' or label == 'unknown':
                    empty_count += 1
                else:
                    # Add accumulated empty squares
                    if empty_count > 0:
                        fen_rank += str(empty_count)
                        empty_count = 0
                    # Add piece
                    fen_rank += reverse_map.get(label, '?')
            
            # Add remaining empty squares
            if empty_count > 0:
                fen_rank += str(empty_count)
            
            fen_ranks.append(fen_rank)
        
        return '/'.join(fen_ranks)
    
    @staticmethod
    def get_piece_classes() -> List[str]:
        """
        Get all possible piece classes (including empty).
        
        Returns:
            List of class names
        """
        classes = ['empty'] + sorted(set(FENParser.PIECE_MAP.values()))
        return classes


def test_extractor():
    """
    Test the square extractor on a sample warped board.
    """
    from board_detector import BoardDetector
    import os
    
    # Path to sample image
    sample_image_path = "/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon/data/per_frame/game2_per_frame/tagged_images/frame_000200.jpg"
    
    if not os.path.exists(sample_image_path):
        print(f"Sample image not found: {sample_image_path}")
        return
    
    # Load and warp image
    image = cv2.imread(sample_image_path)
    detector = BoardDetector(board_size=512)
    warped = detector.detect_board(image, debug=False)
    
    if warped is None:
        print("Failed to detect board")
        return
    
    # Extract squares
    extractor = SquareExtractor(board_size=512)
    squares = extractor.extract_squares(warped)
    
    print(f"Extracted {len(squares)} squares")
    
    # Parse FEN for this frame
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"  # Starting position
    labels = FENParser.fen_to_labels(fen)
    
    print(f"FEN: {fen}")
    print(f"Labels: {labels[:8]}...")  # First rank
    
    # Visualize some squares
    extractor.visualize_squares(squares, labels, num_display=16)
    
    # Test FEN conversion
    reconstructed_fen = FENParser.labels_to_fen(labels)
    print(f"Reconstructed FEN: {reconstructed_fen}")
    print(f"Match: {fen == reconstructed_fen}")
    
    # Print all piece classes
    print(f"\nAll piece classes: {FENParser.get_piece_classes()}")


if __name__ == "__main__":
    test_extractor()

