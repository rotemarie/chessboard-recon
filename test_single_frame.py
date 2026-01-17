"""
Preprocess a single frame for detailed debugging.
"""

import os
import sys
import csv
import cv2
from pathlib import Path

# Add preprocessing to path
sys.path.append(str(Path(__file__).parent / 'preprocessing'))

from preprocessing.board_detector import BoardDetector
from preprocessing.square_extractor import SquareExtractor, FENParser


def process_single_frame():
    """Process frame_000588.jpg and save to test-g2-perframe1."""
    
    # Paths
    project_root = Path(__file__).parent
    frame_num = 588
    
    game2_dir = project_root / "data/per_frame/game2_per_frame"
    csv_file = game2_dir / "game2.csv"
    image_path = game2_dir / "tagged_images" / f"frame_{frame_num:06d}.jpg"
    
    # Output directory
    output_dir = project_root / "test-g2-perframe1"
    squares_dir = output_dir / "squares"
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    squares_dir.mkdir(exist_ok=True)
    
    print(f"Processing single frame: {image_path}")
    print(f"Output directory: {output_dir}")
    
    # Find FEN for this frame
    fen = None
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['from_frame']) == frame_num:
                fen = row['fen']
                break
    
    if fen is None:
        print(f"ERROR: Could not find FEN for frame {frame_num}")
        return
    
    print(f"FEN: {fen}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return
    
    print(f"Image loaded: {image.shape}")
    
    # Save original
    cv2.imwrite(str(output_dir / "00_original.jpg"), image)
    print("✓ Saved: 00_original.jpg")
    
    # Initialize processors
    detector = BoardDetector(board_size=512)
    extractor = SquareExtractor(board_size=512)
    
    # Detect board
    print("\nDetecting board...")
    warped_board = detector.detect_board(image, debug=False)
    
    if warped_board is None:
        print("ERROR: Board detection failed!")
        return
    
    print(f"✓ Board detected and warped: {warped_board.shape}")
    
    # Save warped board
    cv2.imwrite(str(output_dir / "01_warped_board.jpg"), warped_board)
    print("✓ Saved: 01_warped_board.jpg")
    
    # Extract squares
    print("\nExtracting 64 squares...")
    squares = extractor.extract_squares(warped_board)
    print(f"✓ Extracted {len(squares)} squares")
    
    # Parse FEN
    print("\nParsing FEN...")
    labels = FENParser.fen_to_labels(fen)
    print(f"✓ Parsed {len(labels)} labels")
    
    # Create class directories
    piece_classes = FENParser.get_piece_classes()
    for piece_class in piece_classes:
        (squares_dir / piece_class).mkdir(exist_ok=True)
    
    # Save each square with label
    print("\nSaving squares...")
    class_counts = {}
    
    for idx, (square, label) in enumerate(zip(squares, labels)):
        position = extractor.get_square_position(idx)
        filename = f"square_{idx:02d}_{position}_{label}.jpg"
        
        # Save to class folder
        class_path = squares_dir / label / filename
        cv2.imwrite(str(class_path), square)
        
        # Also save to numbered folder for easy viewing
        numbered_path = output_dir / f"square_{idx:02d}_{position}.jpg"
        cv2.imwrite(str(numbered_path), square)
        
        # Count
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print(f"✓ Saved all {len(squares)} squares")
    
    # Create visualization grid
    print("\nCreating visualization grid...")
    grid_img = create_grid_visualization(squares, labels, extractor)
    cv2.imwrite(str(output_dir / "02_grid_visualization.jpg"), grid_img)
    print("✓ Saved: 02_grid_visualization.jpg")
    
    # Save summary
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Frame: {frame_num}\n")
        f.write(f"FEN: {fen}\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"\nOriginal size: {image.shape}\n")
        f.write(f"Warped size: {warped_board.shape}\n")
        f.write(f"Squares extracted: {len(squares)}\n")
        f.write(f"\nClass distribution:\n")
        for label in sorted(class_counts.keys()):
            f.write(f"  {label:20s}: {class_counts[label]:2d} squares\n")
    
    print(f"\n✓ Summary saved to: {summary_file}")
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print("\nFiles created:")
    print("  00_original.jpg           - Original input image")
    print("  01_warped_board.jpg       - Warped 512x512 board")
    print("  02_grid_visualization.jpg - 8x8 grid with labels")
    print("  square_XX_position.jpg    - Individual squares (64 files)")
    print("  squares/<class>/          - Squares organized by piece class")
    print("  summary.txt               - Processing summary")


def create_grid_visualization(squares, labels, extractor):
    """Create an 8x8 grid visualization of all squares with labels."""
    import numpy as np
    
    # Parameters
    square_size = 64
    border = 2
    label_height = 20
    cell_size = square_size + border * 2
    
    # Create canvas
    grid_height = 8 * (cell_size + label_height)
    grid_width = 8 * cell_size
    grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    
    # Place each square
    for idx, (square, label) in enumerate(zip(squares, labels)):
        row = idx // 8
        col = idx % 8
        
        y_start = row * (cell_size + label_height) + border
        x_start = col * cell_size + border
        
        # Place square
        grid[y_start:y_start+square_size, x_start:x_start+square_size] = square
        
        # Add label below
        position = extractor.get_square_position(idx)
        label_y = y_start + square_size + label_height - 5
        label_x = x_start + 5
        
        # Draw text background
        cv2.rectangle(grid, 
                     (x_start, y_start + square_size),
                     (x_start + square_size, y_start + square_size + label_height),
                     (240, 240, 240), -1)
        
        # Draw text
        cv2.putText(grid, f"{position}", (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(grid, label.replace('_', ' ')[:10], (label_x, label_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
    
    return grid


if __name__ == "__main__":
    process_single_frame()

