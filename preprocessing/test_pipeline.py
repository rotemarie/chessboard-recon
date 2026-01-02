"""
Test script for the preprocessing pipeline.

This script tests the pipeline on a few sample images to ensure everything works correctly
before running the full preprocessing.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from board_detector import BoardDetector
from square_extractor import SquareExtractor, FENParser


def test_board_detection():
    """Test board detection on sample images."""
    print("\n" + "="*60)
    print("TEST 1: Board Detection")
    print("="*60)
    
    # Sample image paths
    data_root = Path("/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon/data")
    sample_images = [
        data_root / "per_frame/game2_per_frame/tagged_images/frame_000200.jpg",
        data_root / "per_frame/game2_per_frame/tagged_images/frame_000588.jpg",
        data_root / "per_frame/game2_per_frame/tagged_images/frame_001040.jpg",
    ]
    
    detector = BoardDetector(board_size=512)
    
    success_count = 0
    for img_path in sample_images:
        if not img_path.exists():
            print(f"✗ Image not found: {img_path.name}")
            continue
        
        image = cv2.imread(str(img_path))
        warped = detector.detect_board(image, debug=False)
        
        if warped is not None:
            print(f"✓ Successfully detected board in {img_path.name}")
            success_count += 1
        else:
            print(f"✗ Failed to detect board in {img_path.name}")
    
    print(f"\nResult: {success_count}/{len(sample_images)} successful")
    return success_count == len(sample_images)


def test_square_extraction():
    """Test square extraction."""
    print("\n" + "="*60)
    print("TEST 2: Square Extraction")
    print("="*60)
    
    # Load a sample image
    data_root = Path("/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon/data")
    sample_image = data_root / "per_frame/game2_per_frame/tagged_images/frame_000200.jpg"
    
    if not sample_image.exists():
        print(f"✗ Sample image not found")
        return False
    
    # Detect board
    detector = BoardDetector(board_size=512)
    image = cv2.imread(str(sample_image))
    warped = detector.detect_board(image, debug=False)
    
    if warped is None:
        print("✗ Board detection failed")
        return False
    
    # Extract squares
    extractor = SquareExtractor(board_size=512)
    squares = extractor.extract_squares(warped)
    
    print(f"✓ Extracted {len(squares)} squares")
    
    # Check square properties
    expected_size = 512 // 8
    all_correct_size = all(
        s.shape[0] == expected_size and s.shape[1] == expected_size 
        for s in squares
    )
    
    if all_correct_size:
        print(f"✓ All squares have correct size: {expected_size}x{expected_size}")
    else:
        print(f"✗ Some squares have incorrect size")
        return False
    
    # Test position naming
    positions = [extractor.get_square_position(i) for i in range(64)]
    print(f"✓ First 8 positions: {positions[:8]}")
    print(f"✓ Last 8 positions: {positions[56:]}")
    
    expected_first = ['a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8']
    expected_last = ['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1']
    
    if positions[:8] == expected_first and positions[56:] == expected_last:
        print("✓ Position naming is correct")
    else:
        print("✗ Position naming is incorrect")
        return False
    
    return True


def test_fen_parsing():
    """Test FEN parsing."""
    print("\n" + "="*60)
    print("TEST 3: FEN Parsing")
    print("="*60)
    
    # Test cases
    test_fens = [
        # Starting position
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
        # After e4
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR",
        # Complex position
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R",
    ]
    
    all_passed = True
    
    for i, fen in enumerate(test_fens, 1):
        try:
            # Parse FEN to labels
            labels = FENParser.fen_to_labels(fen)
            
            # Check we got 64 labels
            if len(labels) != 64:
                print(f"✗ Test {i}: Expected 64 labels, got {len(labels)}")
                all_passed = False
                continue
            
            # Convert back to FEN
            reconstructed = FENParser.labels_to_fen(labels)
            
            # Check if it matches
            if fen == reconstructed:
                print(f"✓ Test {i}: FEN parsing correct")
            else:
                print(f"✗ Test {i}: FEN mismatch")
                print(f"  Original:      {fen}")
                print(f"  Reconstructed: {reconstructed}")
                all_passed = False
        
        except Exception as e:
            print(f"✗ Test {i}: Exception - {e}")
            all_passed = False
    
    # Test piece classes
    classes = FENParser.get_piece_classes()
    print(f"\n✓ Found {len(classes)} piece classes:")
    for cls in classes:
        print(f"  - {cls}")
    
    expected_classes = 13  # 12 pieces + empty
    if len(classes) == expected_classes:
        print(f"✓ Correct number of classes ({expected_classes})")
    else:
        print(f"✗ Expected {expected_classes} classes, got {len(classes)}")
        all_passed = False
    
    return all_passed


def test_full_pipeline():
    """Test the full pipeline on one image."""
    print("\n" + "="*60)
    print("TEST 4: Full Pipeline")
    print("="*60)
    
    # Load sample data
    data_root = Path("/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon/data")
    sample_image = data_root / "per_frame/game2_per_frame/tagged_images/frame_000200.jpg"
    sample_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"  # Starting position
    
    if not sample_image.exists():
        print(f"✗ Sample image not found")
        return False
    
    print(f"Processing: {sample_image.name}")
    print(f"FEN: {sample_fen}")
    
    # Step 1: Load image
    image = cv2.imread(str(sample_image))
    print(f"✓ Loaded image: {image.shape}")
    
    # Step 2: Detect and warp board
    detector = BoardDetector(board_size=512)
    warped = detector.detect_board(image, debug=False)
    
    if warped is None:
        print("✗ Board detection failed")
        return False
    print(f"✓ Warped board: {warped.shape}")
    
    # Step 3: Extract squares
    extractor = SquareExtractor(board_size=512)
    squares = extractor.extract_squares(warped)
    print(f"✓ Extracted {len(squares)} squares")
    
    # Step 4: Parse FEN
    labels = FENParser.fen_to_labels(sample_fen)
    print(f"✓ Parsed FEN into {len(labels)} labels")
    
    # Step 5: Verify labels match squares
    if len(squares) == len(labels) == 64:
        print("✓ Counts match: 64 squares and 64 labels")
    else:
        print(f"✗ Count mismatch: {len(squares)} squares, {len(labels)} labels")
        return False
    
    # Step 6: Count pieces
    piece_counts = {}
    for label in labels:
        piece_counts[label] = piece_counts.get(label, 0) + 1
    
    print("\n✓ Piece distribution:")
    for piece, count in sorted(piece_counts.items()):
        print(f"  {piece:15s}: {count:2d}")
    
    # For starting position, we expect specific counts
    expected_counts = {
        'empty': 32,
        'white_pawn': 8,
        'black_pawn': 8,
        'white_rook': 2,
        'black_rook': 2,
        'white_knight': 2,
        'black_knight': 2,
        'white_bishop': 2,
        'black_bishop': 2,
        'white_queen': 1,
        'black_queen': 1,
        'white_king': 1,
        'black_king': 1,
    }
    
    if piece_counts == expected_counts:
        print("\n✓ Piece counts match expected distribution for starting position!")
    else:
        print("\n✗ Piece counts don't match expected distribution")
        return False
    
    return True


def visualize_sample():
    """Create a visualization of the pipeline."""
    print("\n" + "="*60)
    print("Creating visualization...")
    print("="*60)
    
    data_root = Path("/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon/data")
    sample_image = data_root / "per_frame/game2_per_frame/tagged_images/frame_000200.jpg"
    sample_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    
    if not sample_image.exists():
        print("Sample image not found")
        return
    
    # Load and process
    image = cv2.imread(str(sample_image))
    detector = BoardDetector(board_size=512)
    warped = detector.detect_board(image, debug=False)
    
    if warped is None:
        print("Board detection failed")
        return
    
    extractor = SquareExtractor(board_size=512)
    squares = extractor.extract_squares(warped)
    labels = FENParser.fen_to_labels(sample_fen)
    
    # Create visualization
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    
    # Show first 16 squares (top 2 ranks)
    for i in range(16):
        row = i // 8
        col = i % 8
        ax = axes[row, col]
        
        square_rgb = cv2.cvtColor(squares[i], cv2.COLOR_BGR2RGB)
        ax.imshow(square_rgb)
        
        position = extractor.get_square_position(i)
        label = labels[i]
        ax.set_title(f"{position}\n{label}", fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path("/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon/preprocessing/test_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_path}")
    plt.close()


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#" + "  PREPROCESSING PIPELINE TESTS".center(58) + "#")
    print("#" + " "*58 + "#")
    print("#"*60)
    
    tests = [
        ("Board Detection", test_board_detection),
        ("Square Extraction", test_square_extraction),
        ("FEN Parsing", test_fen_parsing),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Create visualization
    try:
        visualize_sample()
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, result in results if result)
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to run full preprocessing.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

