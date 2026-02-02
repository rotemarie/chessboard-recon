"""
Test script for the evaluation API function.
Verifies that predict_board meets all specifications.
"""

import numpy as np
import torch
import cv2
from pathlib import Path

from evaluate import predict_board, CLASS_TO_INT


def test_output_format(board: torch.Tensor) -> bool:
    """Verify the output tensor meets all format requirements."""
    checks = []
    
    # Check 1: Type
    is_tensor = isinstance(board, torch.Tensor)
    checks.append(("Is torch.Tensor", is_tensor))
    
    if not is_tensor:
        return False, checks
    
    # Check 2: Shape
    correct_shape = board.shape == (8, 8)
    checks.append(("Shape is (8, 8)", correct_shape))
    
    # Check 3: Device
    on_cpu = board.device == torch.device('cpu')
    checks.append(("Device is CPU", on_cpu))
    
    # Check 4: Dtype
    correct_dtype = board.dtype in [torch.int64, torch.long]
    checks.append(("Dtype is int64", correct_dtype))
    
    # Check 5: Value range
    valid_values = torch.all((board >= 0) & (board <= 13))
    checks.append(("All values in [0, 13]", valid_values.item()))
    
    all_passed = all(check[1] for check in checks)
    return all_passed, checks


def visualize_board(board: torch.Tensor) -> None:
    """Print a visual representation of the board."""
    reverse_map = {v: k for k, v in CLASS_TO_INT.items()}
    
    # Abbreviated piece symbols
    symbols = {
        0: "♙", 1: "♖", 2: "♘", 3: "♗", 4: "♕", 5: "♔",  # White
        6: "♟", 7: "♜", 8: "♞", 9: "♝", 10: "♛", 11: "♚",  # Black
        12: "·", 13: "?"  # Empty, Unknown
    }
    
    print("\nBoard Visualization (symbols):")
    print("  a b c d e f g h")
    for rank in range(8):
        print(f"{8-rank} ", end="")
        for file in range(8):
            val = board[rank, file].item()
            symbol = symbols.get(val, "?")
            print(symbol, end=" ")
        print(f"{8-rank}")
    print("  a b c d e f g h")
    
    print("\nBoard Values (integers):")
    print(board.numpy())
    
    print("\nClass Distribution:")
    unique, counts = torch.unique(board, return_counts=True)
    for val, count in zip(unique.tolist(), counts.tolist()):
        class_name = reverse_map.get(val, "unknown")
        symbol = symbols.get(val, "?")
        print(f"  {val:2d} {symbol} ({class_name:15s}): {count:2d} squares")


def test_with_image(image_path: str) -> None:
    """Test the evaluation function with a real image."""
    print(f"\n{'='*70}")
    print(f"Testing with image: {image_path}")
    print(f"{'='*70}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image {image_path}")
        return
    
    # Convert to RGB (as required by API)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"\n✓ Loaded image: shape={image_rgb.shape}, dtype={image_rgb.dtype}")
    
    # Run prediction
    try:
        board = predict_board(image_rgb)
        print(f"✓ Prediction completed successfully")
    except Exception as e:
        print(f"❌ Prediction failed with error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Verify output format
    all_passed, checks = test_output_format(board)
    
    print(f"\n{'Output Format Validation':^70}")
    print(f"{'-'*70}")
    for check_name, passed in checks:
        status = "✓" if passed else "❌"
        print(f"{status} {check_name}")
    print(f"{'-'*70}")
    
    if all_passed:
        print(f"✅ All format checks PASSED")
    else:
        print(f"❌ Some format checks FAILED")
    
    # Visualize the board
    visualize_board(board)


def main():
    """Run tests on sample images."""
    import sys
    
    if len(sys.argv) > 1:
        # Test with provided image paths
        for image_path in sys.argv[1:]:
            test_with_image(image_path)
    else:
        # Test with demo images if available
        demo_images = [
            "full_demo/og.jpeg",
            "temp/current_demo_image.jpg",
        ]
        
        found_any = False
        for img_path in demo_images:
            if Path(img_path).exists():
                test_with_image(img_path)
                found_any = True
        
        if not found_any:
            print("No demo images found. Usage:")
            print("  python test_evaluate.py <image_path> [image_path2 ...]")
            print("\nExample:")
            print("  python test_evaluate.py full_demo/og.jpeg")


if __name__ == "__main__":
    main()
