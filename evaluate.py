"""
Evaluation API for Project 1 - Chessboard State Prediction
Implements the required predict_board function according to course specifications.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
import cv2
from PIL import Image
import torchvision
from torchvision import transforms

from preprocessing.board_detector import BoardDetector
from preprocessing.create_block_dataset import BlockSquareExtractor


# Class name to required integer encoding mapping
CLASS_TO_INT = {
    "white_pawn": 0,
    "white_rook": 1,
    "white_knight": 2,
    "white_bishop": 3,
    "white_queen": 4,
    "white_king": 5,
    "black_pawn": 6,
    "black_rook": 7,
    "black_knight": 8,
    "black_bishop": 9,
    "black_queen": 10,
    "black_king": 11,
    "empty": 12,
    "unknown": 13,  # OOD/occluded
}

# our model's class names (alphabetically sorted as in classes.txt)
MODEL_CLASSES = [
    "black_bishop",
    "black_king",
    "black_knight",
    "black_pawn",
    "black_queen",
    "black_rook",
    "empty",
    "white_bishop",
    "white_king",
    "white_knight",
    "white_pawn",
    "white_queen",
    "white_rook",
]


class ChessboardPredictor:
    """Wrapper class for model loading and prediction."""
    
    def __init__(
        self,
        model_path: str = "model/resnet18_ft_blocks_black.pth",
        threshold: float = 0.5,
        board_size: int = 512,
    ):
        self.threshold = threshold
        self.board_size = board_size
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        
        # Setup preprocessing
        self.detector = BoardDetector(board_size=board_size)
        self.block_extractor = BlockSquareExtractor(
            board_size=board_size,
            border_mode="constant",
            border_color="black",
        )
        
        # Setup transform
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the trained ResNet18 model."""
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(MODEL_CLASSES))
        
        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model
    
    def predict(self, image: np.ndarray) -> torch.Tensor:
        """
        Predict the chessboard state from a single RGB image.
        
        Args:
            image: numpy.ndarray of shape (H, W, 3), RGB, uint8, [0, 255]
        
        Returns:
            torch.Tensor of shape (8, 8), dtype int64, on CPU
            Values in range [0, 13] according to class encoding
        """
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detect and warp board
        warped = self.detector.detect_board(image_bgr, debug=False)
        if warped is None:
            # If detection fails, return all unknown
            return torch.full((8, 8), 13, dtype=torch.int64)
        
        # Extract 64 blocks (3x3 context)
        blocks = self.block_extractor.extract_blocks(warped)
        if len(blocks) != 64:
            return torch.full((8, 8), 13, dtype=torch.int64)
        
        # Prepare batch for model
        images = []
        for block in blocks:
            # Convert BGR to RGB for model
            rgb = cv2.cvtColor(block, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            images.append(self.transform(img))
        
        batch = torch.stack(images).to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
        
        # Convert predictions to required format
        confs_cpu = confs.cpu()
        preds_cpu = preds.cpu()
        
        board_state = torch.zeros(64, dtype=torch.int64)
        
        for idx in range(64):
            pred_idx = preds_cpu[idx].item()
            conf = confs_cpu[idx].item()
            
            if conf < self.threshold:
                # Low confidence → OOD/unknown
                board_state[idx] = 13
            else:
                # Map from model class index to required encoding
                class_name = MODEL_CLASSES[pred_idx]
                board_state[idx] = CLASS_TO_INT[class_name]
        
        # Reshape to (8, 8) board
        # Board is indexed in FEN order: a8, b8, ..., h8, a7, ..., h1
        # Output convention: [0,0] = top-left image square, [7,7] = bottom-right
        board_2d = board_state.view(8, 8)
        
        return board_2d


# Global predictor instance (initialized on first call)
_predictor: Optional[ChessboardPredictor] = None


def predict_board(image: np.ndarray) -> torch.Tensor:
    """
    Predict the chessboard state from a single RGB image.
    
    This is the main evaluation function required for Project 1.
    
    Args:
        image: numpy.ndarray
            - Shape: (H, W, 3)
            - Channel order: RGB
            - Dtype: uint8
            - Value range: [0, 255]
    
    Returns:
        torch.Tensor
            - Shape: (8, 8)
            - Device: CPU
            - Dtype: torch.int64
            - Values: [0, 13] according to class encoding:
                0 = White Pawn
                1 = White Rook
                2 = White Knight
                3 = White Bishop
                4 = White Queen
                5 = White King
                6 = Black Pawn
                7 = Black Rook
                8 = Black Knight
                9 = Black Bishop
                10 = Black Queen
                11 = Black King
                12 = Empty Square
                13 = OOD/Unknown
    
    Board Coordinate Convention:
        - output[0, 0] = top-left chess board square of the image
        - output[0, 7] = top-right square
        - output[7, 0] = bottom-left square
        - output[7, 7] = bottom-right square
    """
    global _predictor
    
    # Initialize predictor on first call
    if _predictor is None:
        _predictor = ChessboardPredictor()
    
    return _predictor.predict(image)


def main():
    """Test the evaluation function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Load image as RGB
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        sys.exit(1)
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run prediction
    print(f"Running prediction on {image_path}...")
    board = predict_board(image_rgb)
    
    # Print results
    print(f"\nBoard shape: {board.shape}")
    print(f"Board dtype: {board.dtype}")
    print(f"Board device: {board.device}")
    print(f"\nBoard state (8x8 tensor):")
    print(board)
    
    # Print class distribution
    unique, counts = torch.unique(board, return_counts=True)
    print(f"\nClass distribution:")
    reverse_map = {v: k for k, v in CLASS_TO_INT.items()}
    for val, count in zip(unique.tolist(), counts.tolist()):
        class_name = reverse_map.get(val, "unknown")
        print(f"  {val:2d} ({class_name:15s}): {count:2d} squares")


if __name__ == "__main__":
    main()
