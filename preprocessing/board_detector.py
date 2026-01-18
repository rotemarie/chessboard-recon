"""
Board Detection and Warping Module

This module handles:
1. Finding the chessboard in an image
2. Detecting the four corners of the board
3. Applying perspective transform to get a top-down view
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class BoardDetector:
    """
    Detects chessboards in images and applies perspective transformation.
    """
    
    def __init__(self, board_size: int = 512):
        """
        Initialize the board detector.
        
        Args:
            board_size: Target size for the warped board (will be square)
        """
        self.board_size = board_size
        
    def detect_board(self, image: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
        """
        Detect the chessboard and return a top-down warped view.
        
        Args:
            image: Input image (BGR format from cv2)
            debug: If True, visualize intermediate steps
            
        Returns:
            Warped board image or None if detection failed
        """
        # Find the four corners of the board
        corners = self._find_board_corners(image, debug=debug)
        
        if corners is None:
            print("Failed to detect board corners")
            return None
        
        # Apply perspective transform
        warped = self._warp_board(image, corners)
        
        if debug:
            self._visualize_detection(image, corners, warped)
        
        return warped
    
    def _find_board_corners(self, image: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
        """
        Find the four corners of the chessboard.
        
        Strategy:
        1. Convert to grayscale
        2. Apply edge detection
        3. Find contours
        4. Filter for quadrilaterals (4 corners)
        5. Select the largest quadrilateral that looks like a board
        
        Args:
            image: Input image
            debug: If True, show intermediate steps
            
        Returns:
            4x2 array of corner coordinates [top-left, top-right, bottom-right, bottom-left]
            or None if detection failed
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        # These thresholds may need tuning based on your images
        edges = cv2.Canny(blurred, 50, 150)
        
        if debug:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Original')
            plt.subplot(132)
            plt.imshow(blurred, cmap='gray')
            plt.title('Blurred')
            plt.subplot(133)
            plt.imshow(edges, cmap='gray')
            plt.title('Edges')
            plt.show()
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Try to find a quadrilateral
        board_contour = None
        for contour in contours[:10]:  # Check top 10 largest contours
            # Approximate the contour to a polygon
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # Check if it's a quadrilateral and sufficiently large
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                # Board should be at least 20% of image area
                if area > 0.2 * image.shape[0] * image.shape[1]:
                    board_contour = approx
                    break
        
        if board_contour is None:
            # Fallback: Try a different approach using morphological operations
            return self._find_corners_alternative(image, debug)
        
        # Reshape to (4, 2) array
        corners = board_contour.reshape(4, 2)
        
        # Order the corners: top-left, top-right, bottom-right, bottom-left
        corners = self._order_corners(corners)
        
        return corners
    
    def _find_corners_alternative(self, image: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
        """
        Alternative corner detection method using adaptive thresholding and morphology.
        
        This method works better when the board has strong grid lines.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding to handle lighting variations
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to enhance board structure
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        if debug:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.imshow(thresh, cmap='gray')
            plt.title('Adaptive Threshold')
            plt.subplot(122)
            plt.imshow(morph, cmap='gray')
            plt.title('Morphology')
            plt.show()
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box and expand slightly
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Create corners from bounding box with small margin
        margin = 10
        corners = np.array([
            [x - margin, y - margin],  # top-left
            [x + w + margin, y - margin],  # top-right
            [x + w + margin, y + h + margin],  # bottom-right
            [x - margin, y + h + margin]  # bottom-left
        ], dtype=np.float32)
        
        # Ensure corners are within image bounds
        corners[:, 0] = np.clip(corners[:, 0], 0, image.shape[1] - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, image.shape[0] - 1)
        
        return corners
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners as: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            corners: 4x2 array of corner coordinates
            
        Returns:
            Ordered 4x2 array
        """
        # Initialize ordered corners array
        ordered = np.zeros((4, 2), dtype=np.float32)
        
        # Sum of coordinates: top-left has smallest sum, bottom-right has largest
        s = corners.sum(axis=1)
        ordered[0] = corners[np.argmin(s)]  # top-left
        ordered[2] = corners[np.argmax(s)]  # bottom-right
        
        # Difference of coordinates: top-right has smallest diff, bottom-left has largest
        diff = np.diff(corners, axis=1)
        ordered[1] = corners[np.argmin(diff)]  # top-right
        ordered[3] = corners[np.argmax(diff)]  # bottom-left
        
        return ordered
    
    def _warp_board(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Apply perspective transform to get a top-down view of the board.
        
        Args:
            image: Input image
            corners: 4x2 array of corner coordinates (ordered)
            
        Returns:
            Warped board image (square, top-down view)
        """
        # Define destination points (perfect square)
        dst = np.array([
            [0, 0],
            [self.board_size - 1, 0],
            [self.board_size - 1, self.board_size - 1],
            [0, self.board_size - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        
        # Apply the transform
        warped = cv2.warpPerspective(image, matrix, (self.board_size, self.board_size))
        
        return warped
    
    def _visualize_detection(self, original: np.ndarray, corners: np.ndarray, 
                            warped: np.ndarray) -> None:
        """
        Visualize the detection results.
        """
        # Draw corners on original image
        img_with_corners = original.copy()
        for i, corner in enumerate(corners):
            cv2.circle(img_with_corners, tuple(corner.astype(int)), 10, (0, 255, 0), -1)
            cv2.putText(img_with_corners, str(i), tuple(corner.astype(int)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Draw lines connecting corners
        for i in range(4):
            pt1 = tuple(corners[i].astype(int))
            pt2 = tuple(corners[(i + 1) % 4].astype(int))
            cv2.line(img_with_corners, pt1, pt2, (0, 255, 0), 2)
        
        # Display results
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 5))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
        plt.title('Detected Board Corners')
        plt.subplot(122)
        plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        plt.title('Warped Board (Top-Down)')
        plt.show()


def test_detector():
    """
    Test the board detector on a sample image.
    """
    import os
    
    # Path to a sample image
    sample_image_path = "/Users/rotemar/Documents/BGU/Intro to Deep Learning/final project/chessboard-recon/data/per_frame/game2_per_frame/tagged_images/frame_000200.jpg"
    
    if not os.path.exists(sample_image_path):
        print(f"Sample image not found: {sample_image_path}")
        return
    
    # Load image
    image = cv2.imread(sample_image_path)
    print(f"Loaded image with shape: {image.shape}")
    
    # Create detector
    detector = BoardDetector(board_size=512)
    
    # Detect and warp board
    warped = detector.detect_board(image, debug=True)
    
    if warped is not None:
        print(f"Successfully warped board to shape: {warped.shape}")
    else:
        print("Board detection failed")


if __name__ == "__main__":
    test_detector()

