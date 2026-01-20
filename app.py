"""
Chessboard Recognition Demo App
Interactive UI for project presentation
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import sys
from PIL import Image
import json

# Add preprocessing to path
sys.path.append(str(Path(__file__).parent / 'preprocessing'))
sys.path.append(str(Path(__file__).parent / 'inference'))

from preprocessing.board_detector import BoardDetector
from preprocessing.square_extractor import SquareExtractor, FENParser

# Try to import inference pipeline (lazy import to avoid permission errors)
INFERENCE_AVAILABLE = False
inference_modules = {}

def load_inference_modules():
    """Lazy load inference modules to avoid permission errors on startup."""
    global INFERENCE_AVAILABLE, inference_modules
    if not INFERENCE_AVAILABLE and not inference_modules:
        try:
            import torch
            from inference.pipeline import (
                load_model, 
                make_transform, 
                classify_squares,
                labels_to_fen_with_unknown,
                render_board_svg,
                load_class_names,
                run_pipeline
            )
            inference_modules = {
                'torch': torch,
                'load_model': load_model,
                'make_transform': make_transform,
                'classify_squares': classify_squares,
                'labels_to_fen_with_unknown': labels_to_fen_with_unknown,
                'render_board_svg': render_board_svg,
                'load_class_names': load_class_names,
                'run_pipeline': run_pipeline
            }
            INFERENCE_AVAILABLE = True
            return True
        except Exception as e:
            st.error(f"Failed to load inference modules: {e}")
            return False
    return INFERENCE_AVAILABLE


# Page configuration
st.set_page_config(
    page_title="Chess Board Recognition",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #A23B72;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .stButton>button {
        width: 100%;
        background-color: #2E86AB;
        color: white;
        font-weight: bold;
    }
    .chess-square {
        border: 1px solid #ddd;
        padding: 2px;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'warped_board' not in st.session_state:
        st.session_state.warped_board = None
    if 'squares' not in st.session_state:
        st.session_state.squares = None
    if 'labels' not in st.session_state:
        st.session_state.labels = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'fen' not in st.session_state:
        st.session_state.fen = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'class_names' not in st.session_state:
        st.session_state.class_names = None
    if 'board_svg' not in st.session_state:
        st.session_state.board_svg = None
    if 'confidences' not in st.session_state:
        st.session_state.confidences = None
    if 'unknown_indices' not in st.session_state:
        st.session_state.unknown_indices = None


def load_sample_images():
    """Load available sample images."""
    samples_dir = Path(__file__).parent / "data/per_frame/game2_per_frame/tagged_images"
    if samples_dir.exists():
        return sorted(list(samples_dir.glob("frame_*.jpg")))[:10]  # First 10 frames
    return []


def run_inference_pipeline(image_path, model_path=None, threshold=0.80):
    """
    Run the complete inference pipeline on an image using pipeline.py.
    
    Args:
        image_path: Path to input image
        model_path: Path to model checkpoint (optional)
        threshold: Confidence threshold for OOD detection
    
    Returns:
        dict with results or None if failed
    """
    # Load inference modules if needed
    if not load_inference_modules():
        return {"error": "Inference modules not available. Please install torch, torchvision, and python-chess."}
    
    try:
        # Check if model exists
        if not model_path or not Path(model_path).exists():
            return {"error": f"Model not found at {model_path}. Please provide a valid model path."}
        
        # Get inference function from pipeline.py
        run_pipeline = inference_modules.get('run_pipeline')
        if not run_pipeline:
            # Import it if not already loaded
            from inference.pipeline import run_pipeline as pipeline_run
            inference_modules['run_pipeline'] = pipeline_run
            run_pipeline = pipeline_run
        
        # Create temporary output directory
        temp_output = Path(__file__).parent / "temp" / "inference_output"
        temp_output.mkdir(parents=True, exist_ok=True)
        
        # Setup paths
        project_root = Path(__file__).parent
        class_dir = project_root / "dataset" / "train"
        classes_file = project_root / "model" / "classes.txt"
        
        # Run the complete pipeline from pipeline.py
        run_pipeline(
            image_path=str(image_path),
            model_path=str(model_path),
            output_dir=str(temp_output),
            class_dir=str(class_dir) if class_dir.exists() else None,
            classes_file=str(classes_file) if classes_file.exists() else None,
            threshold=threshold,
            board_size=512,
            render_size=512,
            save_square_crops=False,
            print_squares=False,
            crops_dir=None,
            save_grid=True
        )
        
        # Read the results
        original = cv2.imread(str(image_path))
        warped = cv2.imread(str(temp_output / "warped_board.jpg"))
        fen_path = temp_output / "fen.txt"
        svg_path = temp_output / "board.svg"
        predictions_path = temp_output / "predictions.json"
        grid_path = temp_output / "crops_grid.jpg"
        
        fen = fen_path.read_text(encoding="utf-8") if fen_path.exists() else None
        board_svg = svg_path.read_text(encoding="utf-8") if svg_path.exists() else None
        
        # Load predictions
        predictions = []
        labels = []
        confidences = []
        unknown_indices = []
        
        if predictions_path.exists():
            predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
            labels = [p["label"] for p in predictions]
            confidences = [p["confidence"] for p in predictions]
            unknown_indices = [i for i, p in enumerate(predictions) if p["label"] == "unknown"]
        
        return {
            "success": True,
            "original": original,
            "warped": warped,
            "grid": cv2.imread(str(grid_path)) if grid_path.exists() else None,
            "labels": labels,
            "confidences": confidences,
            "unknown_indices": unknown_indices,
            "fen": fen,
            "board_svg": board_svg,
            "predictions": predictions
        }
    
    except Exception as e:
        import traceback
        return {"error": f"{str(e)}\n\n{traceback.format_exc()}"}


def preprocess_image(image_array, show_debug=False):
    """Preprocess the chess board image."""
    # Initialize processors
    detector = BoardDetector(board_size=512)
    extractor = SquareExtractor(board_size=512)
    
    # Detect and warp board
    with st.spinner("Detecting chessboard..."):
        warped_board = detector.detect_board(image_array, debug=show_debug)
    
    if warped_board is None:
        return None, None
    
    # Extract squares
    with st.spinner("Extracting 64 squares..."):
        squares = extractor.extract_squares(warped_board)
    
    return warped_board, squares


def create_grid_image(squares, labels=None, predictions=None):
    """Create an 8x8 grid visualization of squares."""
    if squares is None:
        return None
    
    # Parameters
    square_size = 64
    border = 2
    label_height = 30
    cell_size = square_size + border * 2
    
    # Create canvas
    grid_height = 8 * (cell_size + label_height)
    grid_width = 8 * cell_size
    grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    
    # Place each square
    for idx, square in enumerate(squares):
        row = idx // 8
        col = idx % 8
        
        y_start = row * (cell_size + label_height) + border
        x_start = col * cell_size + border
        
        # Place square
        grid[y_start:y_start+square_size, x_start:x_start+square_size] = square
        
        # Add label below
        file_letter = chr(ord('a') + col)
        rank_number = 8 - row
        position = f"{file_letter}{rank_number}"
        
        label_y = y_start + square_size + 20
        label_x = x_start + 10
        
        # Draw background
        cv2.rectangle(grid, 
                     (x_start, y_start + square_size),
                     (x_start + square_size, y_start + square_size + label_height),
                     (245, 245, 245), -1)
        
        # Draw position
        cv2.putText(grid, position, (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw label or prediction
        if labels and idx < len(labels):
            label_text = labels[idx].replace('_', ' ')[:10]
            cv2.putText(grid, label_text, (label_x, label_y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        elif predictions and idx < len(predictions):
            pred_text = predictions[idx][:10]
            color = (0, 150, 0) if pred_text != "unknown" else (200, 0, 0)
            cv2.putText(grid, pred_text, (label_x, label_y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    return grid


def main():
    """Main application."""
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header">Chessboard Recognition System</div>', 
                unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #666; margin-bottom: 0.5rem;">'
                'Deep Learning Project - Ben-Gurion University 2026</div>', 
                unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #888; margin-bottom: 2rem; font-size: 0.9rem;">'
                'Shon Grinberg • David Paster • Rotem Arie</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/ChessSet.jpg/300px-ChessSet.jpg", 
                 width='stretch')
        
        st.markdown("## Project Overview")
        st.markdown("""
        This system uses deep learning to:
        1. **Detect** chessboards in images
        2. **Warp** to top-down view
        3. **Extract** 64 individual squares
        4. **Classify** each piece (13 classes)
        5. **Detect** occlusions (OOD)
        6. **Reconstruct** board in FEN notation
        """)
        
        st.markdown("---")
        st.markdown("## Key Statistics")
        st.metric("Training Accuracy", "99.15%")
        st.metric("Validation Accuracy", "89.08%")
        st.metric("Board Detection Rate", "92.1%")
        
        st.markdown("---")
        st.markdown("## Model Architecture")
        st.markdown("""
        - **Model:** ResNet18 (fine-tuned)
        - **Input:** 224×224 RGB
        - **Classes:** 13 (12 pieces + empty)
        - **OOD:** Confidence thresholding
        """)
        
        st.markdown("---")
        st.markdown("## Project Repository")
        st.markdown("""
        [View on GitHub](https://github.com/rotemarie/chessboard-recon)
        
        Complete source code, documentation, and implementation details.
        """)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "Project Overview", "Pipeline", "Full Demo", "Live Demo"
    ])
    
    # Tab 1: Project Overview
    with tab1:
        st.markdown('<div class="sub-header">Chessboard Square Classification and Board-State Reconstruction</div>', 
                    unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("## Problem Statement")
            st.markdown("""
            Given a single image of a physical chessboard captured from an arbitrary angle, 
            our system must:
            
            1. **Detect and localize** the chessboard within the image
            2. **Extract** all 64 individual squares
            3. **Classify** each square into 13 classes:
               - 12 piece types (white/black × pawn/knight/bishop/rook/queen/king)
               - Empty squares
            4. **Handle occlusions** - detect and mark ambiguous squares as "unknown"
            5. **Reconstruct** the complete board state in FEN notation
            """)
            
            st.markdown("## Key Challenges")
            st.markdown("""
            - **Arbitrary camera angles** - boards photographed from various perspectives
            - **Varying lighting conditions** - shadows, highlights, different environments
            - **Occlusions** - hands, other pieces, or objects blocking the view
            - **Piece similarity** - distinguishing between similar pieces (e.g., bishop vs pawn)
            - **No temporal information** - single static images only (no video context)
            """)
        
        with col2:
            st.markdown("## Our Solution")
            st.info("""
            **Two-Stage Pipeline:**
            
            **Stage 1: Preprocessing**
            - Board detection using edge detection
            - Perspective transformation to top-down view
            - Square extraction (64 images)
            
            **Stage 2: Classification**
            - ResNet18 CNN (fine-tuned on ImageNet)
            - Class balancing via weighted sampling
            - OOD detection via confidence thresholding
            """)
            
            st.markdown("## Dataset")
            st.markdown("""
            - **5 labeled games** (517 frames)
            - **~30,000 labeled squares** after preprocessing
            - **Train/Val/Test split** by game (prevents data leakage)
            - **13 classes** with natural imbalance
            """)
            
            st.markdown("## Results Summary")
            st.success("""
            **Preprocessing:** 92.1% board detection success
            
            **Classification:** 89.08% validation accuracy
            
            **OOD Detection:** 85.4% true positive rate on occluded pieces
            """)
            
            st.markdown("## GitHub Repository")
            st.markdown("""
            Full source code, documentation, and implementation:
            
            **[github.com/rotemarie/chessboard-recon](https://github.com/rotemarie/chessboard-recon)**
            """)
    
    # Tab 2: Pipeline (with nested tabs)
    with tab2:
        st.markdown('<div class="sub-header">Complete Processing Pipeline</div>', 
                    unsafe_allow_html=True)
        
        # Create nested tabs for pipeline stages
        pipeline_tab1, pipeline_tab2, pipeline_tab3 = st.tabs([
            "🔄 Preprocessing", "🧠 Training & OOD", "🔗 Integration"
        ])
        
        # Nested Tab 1: Preprocessing
        with pipeline_tab1:
            st.markdown("### Data Collection and Preprocessing")
            
            st.markdown("---")
            
            # Section 1: Data Collection
            st.markdown("#### 1. Data Collection and Organization")
        
            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown("""
                **Input Data:**
                - 5 chess games with labeled frames (games 2, 4-7)
                - Each frame has a corresponding FEN notation
                - Images captured at various game stages
                - Total: 517 labeled frames
                
                **Data Format:**
                - CSV files with frame numbers and FEN strings
                - JPEG images (480×480 or similar)
                """)
            
            with col2:
                st.code("""
from_frame,to_frame,fen
200,200,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
588,588,rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR
...
                """, language="csv")
            
            st.markdown("---")
            
            # Section 2: Preprocessing Methods
            st.markdown("#### 2. Preprocessing Methods Explored")
            
            st.markdown("""
            We experimented with multiple preprocessing approaches to find the optimal method for square extraction:
            """)
            
            # Show preprocessing comparison image
            preprocessing_img_path = Path(__file__).parent / "for_ui" / "preprocessing.png"
            if preprocessing_img_path.exists():
                st.image(str(preprocessing_img_path), 
                         caption="Comparison of Different Preprocessing Methods", 
                         width=700)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("##### Original (64×64)")
                st.markdown("""
                **Exact square extraction**
                - No padding
                - 64×64 pixels per square
                - **Validation Accuracy: 91.46%**
                
                **Problem:** Pieces extending beyond square boundaries get cut off 
                (especially tall pieces like kings and queens).
                """)
            
            with col2:
                st.markdown("##### 3×3 Blocks")
                st.markdown("""
                **3×3 square neighborhood**
                - 192×192 pixels (3 squares)
                - Black/mirrored padding at edges
                - **Best Validation Accuracy: 92.51%** ✓
                
                **Advantage:** Provides rich spatial context. Each square sees its 
                8 neighbors, helping distinguish similar pieces.
                """)
            
            with col3:
                st.markdown("##### Padded Squares")
                st.markdown("""
                **Single square with padding**
                - 64×64 base + padding (20-30%)
                - Black border padding
                - **Validation Accuracy: 90.36% (30% padding)**
                
                **Advantage:** Captures piece tops without excessive context. 
                Good balance between context and focus.
                """)
            
            st.success("""
            **Final Choice:** 3×3 Blocks with black padding achieved the best results (92.51% validation accuracy), 
            providing optimal spatial context for piece classification.
            """)
            
            st.markdown("---")
            
            # Section 3: Preprocessing Pipeline
            st.markdown("#### 3. Preprocessing Pipeline Implementation")
        
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**A. Board Detection**")
                st.markdown("""
                **Primary Method:** Edge Detection
                1. Convert to grayscale
                2. Gaussian blur (σ=5)
                3. Canny edge detection (50, 150)
                4. Find contours, filter for quadrilaterals
                5. Select largest contour (>20% of image)
                
                **Fallback Method:** Adaptive Thresholding
                - Used when edge detection fails
                - Morphological operations
                - Bounding box estimation
                
                **Success Rate:** 92.1% across all games
                """)
                
                st.markdown("**B. Perspective Transformation**")
                st.markdown("""
                1. **Order corners** consistently (TL, TR, BR, BL)
                2. **Apply homography** to map to 512×512 square
                3. **Result:** Perfect top-down view
                """)
            
            with col2:
                st.markdown("**C. 3×3 Block Extraction**")
                st.markdown("""
                1. **Pad board** with 64px black border on all sides
                2. **For each square** (64 total):
                   - Extract 3×3 neighborhood (192×192 pixels)
                   - Center the target square
                   - Include surrounding context
                3. **Order** following FEN convention (a8 → h1)
                
                **Output:** 64 blocks of 192×192 pixels each
                """)
                
                st.markdown("**D. FEN Parsing and Labeling**")
                st.markdown("""
                1. **Parse** FEN string to 64 labels
                2. **Map** each character to piece class
                3. **Expand** numbers (e.g., '8' → 8 empty squares)
                4. **Validate** total count = 64
                
                **Classes:** 13 total (12 pieces + empty)
                """)
            
            st.markdown("---")
            
            # Section 4: Dataset Preparation
            st.markdown("#### 4. Dataset Preparation")
        
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Dataset Splitting**")
                st.markdown("""
                **Critical:** Split by **game**, not frame!
                
                - **Train:** games 4, 5, 6 (74.5%)
                - **Val:** game 7 (10.6%)
                - **Test:** game 2 (14.9%)
                - **Total:** 33,092 blocks
                
                **Why by game?**
                Prevents data leakage - frames from the same game are highly correlated.
                """)
            
            with col2:
                st.markdown("**Class Balancing**")
                st.markdown("""
                **Problem:** Natural class imbalance
                - Empty: ~69%
                - Pawns: ~16%
                - Other pieces: ~15%
                
                **Solution:** Weighted random sampling
                - Inverse frequency weights
                - Sample with replacement
                - Equal class representation per epoch
                """)
            
            with col3:
                st.markdown("**Data Augmentation**")
                st.markdown("""
                **Training augmentation:**
                - Random horizontal flips
                - Random vertical flips
                - Resize to 224×224
                - ImageNet normalization
                
                **Validation/Test:**
                - No augmentation
                - Resize + normalize only
                """)
        
        # Nested Tab 2: Training & OOD
        with pipeline_tab2:
            st.markdown("### Model Training and OOD Detection")
            
            st.markdown("---")
            
            # Section 1: Model Training
            st.markdown("#### 1. Model Architecture and Training")
        
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.markdown("**Model Selection**")
                st.markdown("""
                **Tested architectures:**
                - ResNet18 (11M params) ✓ **Best**
                - ResNet50 (23M params)
                - VGG16 (138M params)
                
                **Training modes:**
                - **Fine-tuning:** Train all layers (chosen)
                - **Transfer learning:** Freeze backbone, train final layer only
                
                **Why ResNet18?**
                - Best accuracy with 3×3 blocks (92.51%)
                - Reasonable training time
                - Good speed-accuracy tradeoff
                """)
            
            with col2:
                st.markdown("**Training Configuration**")
                st.code("""
# Hyperparameters
Model: ResNet18 (fine-tuned from ImageNet)
Input: 192×192 (3×3 blocks)
Optimizer: SGD (lr=0.001, momentum=0.9)
Scheduler: StepLR (step_size=7, gamma=0.1)
Batch size: 16
Loss: Cross-entropy
Early stopping: patience=10

# Training results (3×3 blocks with black padding)
Epochs to convergence: ~15-20
Training accuracy: 99.15%
Validation accuracy: 92.51% ✓
Training time: ~2-3 hours (GPU)
                """, language="python")
            
            st.markdown("---")
            
            # Section 2: OOD Detection
            st.markdown("#### 2. Out-of-Distribution (OOD) Detection")
        
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Problem: Occlusions**")
                st.markdown("""
                **Observation:** ~13% of validation errors due to occlusions
                - Hands covering pieces
                - Other pieces blocking view
                - Poor lighting/shadows
                
                **Challenge:** Model outputs high-confidence wrong predictions
                
                **Solution:** Confidence-based OOD detection
                """)
                
                st.markdown("**Method: Maximum Softmax Probability**")
                st.markdown("""
                1. Compute softmax probabilities
                2. Take maximum probability as confidence score
                3. If confidence < threshold → mark as "unknown"
                
                **Threshold selection:** 0.80
                - Based on clean vs occluded distribution analysis
                - 5th percentile of clean confidence
                """)
            
            with col2:
                st.markdown("**Results**")
                st.markdown("""
                **Clean images:**
                - Mean confidence: 0.94 ± 0.08
                - False positive rate: 4.8%
                
                **Occluded images:**
                - Mean confidence: 0.62 ± 0.21
                - True positive rate: 85.4%
                
                **Confidence separation:** 0.32
                
                → Clear separation validates the approach!
                """)
                
                st.info("""
                **In practice:**
                - Model predicts normally
                - Low-confidence predictions → "?"
                - FEN output includes unknowns
                - User can manually verify ambiguous squares
                """)
        
        # Nested Tab 3: Integration
        with pipeline_tab3:
            st.markdown("### Board Reconstruction and Integration")
            
            st.markdown("---")
            
            # Section 1: Final Pipeline
            st.markdown("#### 1. Complete Pipeline")
            
            st.markdown("""
            **End-to-End Process:**
            
            ```
            Input Image → Board Detection → Perspective Transform → 3×3 Block Extraction
                                                                              ↓
            FEN Output ← FEN Generator ← OOD Detection ← ResNet18 Classification ← 64 Blocks
            ```
            
            **Output Format:** FEN notation with optional '?' for unknowns
            - Standard: `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR`
            - With unknowns: `rnbqkbnr/pppp?ppp/8/8/8/8/PPPPPPPP/RNBQKBNR`
            """)
            
            st.markdown("---")
            
            # Section 2: Performance Metrics
            st.markdown("#### 2. System Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Overall Metrics**")
                st.markdown("""
                - **Validation Accuracy:** 92.51%
                - **Empty squares:** 95.2%
                - **Kings/Queens:** ~93%
                - **Pawns:** ~87%
                - **Board detection:** 92.1%
                """)
            
            with col2:
                st.markdown("**Error Analysis**")
                st.markdown("""
                **Main error sources:**
                - Similar pieces: 35%
                - Piece cropping: 27%
                - Lighting: 15%
                - Occlusions: 13%
                - Other: 10%
                """)
            
            with col3:
                st.markdown("**Deployment**")
                st.markdown("""
                **Output:**
                - FEN notation
                - Reconstructed board (SVG)
                - 64 classified squares (grid)
                - Confidence scores
                - Unknown markers (red X)
                """)
            
            st.markdown("---")
            
            # Section 3: Usage
            st.markdown("#### 3. How to Use")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Command Line:**")
                st.code("""
python inference/pipeline.py \\
    --image path/to/board.jpg \\
    --model model/resnet18_ft_blocks_black.pth \\
    --classes-file model/classes.txt \\
    --threshold 0.80 \\
    --output-dir outputs/ \\
    --save-grid
                """, language="bash")
            
            with col2:
                st.markdown("**Python API:**")
                st.code("""
from inference.pipeline import run_pipeline

run_pipeline(
    image_path="board.jpg",
    model_path="model/resnet18_ft_blocks_black.pth",
    classes_file="model/classes.txt",
    threshold=0.80,
    save_grid=True
)
                """, language="python")
            
            st.markdown("---")
            
            # Section 4: Future Work
            st.markdown("#### 4. Future Improvements")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Model Enhancements:**")
                st.markdown("""
                - Ensemble methods for higher accuracy
                - Attention mechanisms for piece focus
                - Advanced OOD detection (Mahalanobis distance, energy-based)
                - Domain adaptation for different board styles
                """)
            
            with col2:
                st.markdown("**System Extensions:**")
                st.markdown("""
                - Temporal modeling for video streams
                - Real-time processing optimization
                - Mobile deployment
                - Chess rule validation (legal moves)
                """)
    
    # Tab 3: Input (renamed)    
    # Tab 3: Full Demo (Interactive Walkthrough)
    with tab3:
        st.markdown('<div class="sub-header">Complete Pipeline Walkthrough</div>', 
                    unsafe_allow_html=True)
        
        st.markdown("""
        This interactive demo walks through the complete process from input image to final FEN output.
        Use the controls below to navigate through each stage.
        """)
        
        # Initialize step state
        if 'demo_step' not in st.session_state:
            st.session_state.demo_step = 0
        
        # Define pipeline steps
        steps = [
            {
                "name": "Input",
                "file": "preprocessed.jpeg",
                "title": "Step 1: Input Image",
                "description": """
                **Input:** Raw photo of a physical chessboard taken from an arbitrary angle.
                
                **Challenges:**
                - Perspective distortion
                - Varying lighting conditions
                - Background clutter
                - Different camera angles
                - Pieces at various heights
                
                **Goal:** Transform this raw image into a format suitable for piece classification.
                """
            },
            {
                "name": "Preprocessing",
                "file": "processed.jpeg",
                "title": "Step 2: Preprocessing Pipeline",
                "description": """
                **Board Localization & Warping:**
                1. **Edge Detection:** Canny edge detector finds board boundaries
                2. **Contour Detection:** Identify quadrilateral shapes
                3. **Corner Detection:** Find the four corners of the chessboard
                4. **Perspective Transform:** Warp to 512×512 top-down view
                
                **3×3 Block Extraction:**
                1. **Pad Board:** Add 64px black border around the entire board
                2. **Extract Blocks:** For each of the 64 squares, extract a 3×3 neighborhood (192×192 pixels)
                3. **Center Target Square:** Each block is centered on its target square, including 8 surrounding squares
                4. **Chess Notation:** Label each block by its center square (a8-h1)
                
                **Output:** 64 blocks (192×192 pixels each) ready for classification.
                
                **Key Insight:** This visualization shows all 64 extracted blocks with their 
                chess notation labels. The 3×3 context helps the model distinguish similar pieces by 
                seeing neighboring squares.
                """
            },
            {
                "name": "Classification",
                "file": "board.jpeg",
                "title": "Step 3: Model Classification",
                "description": """
                **Model Architecture:**
                - **Base:** ResNet18 (pre-trained on ImageNet)
                - **Fine-tuning:** Trained on our chess piece dataset
                - **Output:** 13 classes (6 white pieces + 6 black pieces + empty)
                
                **Training Details:**
                - **Dataset:** ~33K square images from 5 games
                - **Split:** Train (70%), Val (15%), Test (15%) - split by game
                - **Augmentation:** Random rotation, brightness, contrast
                - **Loss:** Cross-entropy with weighted sampling for class balance
                - **Optimizer:** Adam with learning rate scheduling
                - **Accuracy:** 89.08% overall, 95.2% on empty squares
                
                **OOD Detection (Out-of-Distribution):**
                - **Method:** Maximum Softmax Probability (MSP)
                - **Threshold:** 0.80 (confidence below → "unknown")
                - **Purpose:** Detect occluded/uncertain squares
                - **Result:** 85.4% true positive rate on occluded squares
                
                **Output:** The visualization is rendered from the FEN string derived from 
                model predictions. Unknown/occluded squares are marked with red X.
                """
            },
            {
                "name": "FEN Output",
                "file": "fen.jpeg",
                "title": "Step 4: FEN Reconstruction & Integration",
                "description": """
                **FEN Generation:**
                1. **Map predictions** to FEN characters:
                   - White pieces: P, N, B, R, Q, K
                   - Black pieces: p, n, b, r, q, k
                   - Empty squares: counted and compressed (e.g., "3" = 3 empty)
                   - Unknown squares: "?" (occluded/low confidence)
                
                2. **Build FEN string:**
                   - Process rank-by-rank from rank 8 → rank 1
                   - Separate ranks with "/"
                   - Compress consecutive empty squares
                
                3. **Add metadata** (if needed):
                   - Active color (w/b)
                   - Castling rights (KQkq)
                   - En passant target
                   - Halfmove/fullmove counters
                
                **Integration:**
                This complete pipeline can be integrated into:
                - Chess analysis tools
                - Game digitization systems
                - Live streaming overlays
                - Tournament recording systems
                
                **Example FEN:** `rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR`
                
                **Final Output:** The reconstructed board can be imported directly into 
                chess engines (Stockfish, Lichess, Chess.com) for analysis and play.
                """
            }
        ]
        
        # Progress bar
        progress = (st.session_state.demo_step + 1) / len(steps)
        st.progress(progress)
        
        # Step indicator
        step_cols = st.columns(len(steps))
        for i, step in enumerate(steps):
            with step_cols[i]:
                if i == st.session_state.demo_step:
                    st.markdown(f"**→ {step['name']}**")
                elif i < st.session_state.demo_step:
                    st.markdown(f"✓ {step['name']}")
                else:
                    st.markdown(f"○ {step['name']}")
        
        st.markdown("---")
        
        # Current step content
        current_step = steps[st.session_state.demo_step]
        
        st.markdown(f"### {current_step['title']}")
        
        # Display based on step (some steps show 2 images)
        if current_step['name'] in ["Classification", "FEN Output"]:
            # For classification and output steps, show 2 images side by side
            st.markdown(current_step['description'])
            
            st.markdown("---")
            st.markdown("#### Outputs")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Reconstructed Board**")
                output_dir = Path(__file__).parent / "output"
                board_path = output_dir / "board.jpeg"
                
                if board_path.exists():
                    image = Image.open(board_path)
                    st.image(image, width=400)
                else:
                    st.info("Board visualization")
            
            with col2:
                st.markdown("**Classified Squares (8×8 Grid)**")
                fen_path = output_dir / "fen.jpeg"
                
                if fen_path.exists():
                    image = Image.open(fen_path)
                    st.image(image, width=400)
                else:
                    st.info("Grid visualization")
        else:
            # For other steps, show single image on left, description on right
            col1, col2 = st.columns([2, 3])
            
            with col1:
                # Load and display image (smaller)
                output_dir = Path(__file__).parent / "output"
                image_path = output_dir / current_step['file']
                
                if image_path.exists():
                    image = Image.open(image_path)
                    st.image(image, width=400)  # Fixed width instead of full container
                else:
                    st.error(f"Image not found: {current_step['file']}")
                    st.info(f"Expected path: {image_path}")
            
            with col2:
                st.markdown(current_step['description'])
        
        st.markdown("---")
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.session_state.demo_step > 0:
                if st.button("← Previous", use_container_width=True):
                    st.session_state.demo_step -= 1
                    st.rerun()
        
        with col2:
            if st.button("🔄 Reset to Start", use_container_width=True):
                st.session_state.demo_step = 0
                st.rerun()
        
        with col3:
            if st.session_state.demo_step < len(steps) - 1:
                if st.button("Next →", use_container_width=True, type="primary"):
                    st.session_state.demo_step += 1
                    st.rerun()
            else:
                st.success("✓ Demo Complete!")
        
        # Quick jump
        st.markdown("---")
        st.markdown("**Quick Jump:**")
        jump_cols = st.columns(len(steps))
        for i, step in enumerate(steps):
            with jump_cols[i]:
                if st.button(f"{i+1}. {step['name']}", key=f"jump_{i}", use_container_width=True):
                    st.session_state.demo_step = i
                    st.rerun()
        
        # Live Inference Section
        st.markdown("---")
        st.markdown("### 🚀 Try Live Inference")
        
        st.markdown("""
        Upload your own chessboard image and run the complete pipeline with a trained model!
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload Chessboard Image", 
                type=['jpg', 'jpeg', 'png'],
                key="live_inference_upload"
            )
        
        with col2:
            model_path = st.text_input(
                "Model Path (optional)",
                value="model/resnet18_ft_blocks_black.pth",
                help="Path to trained model checkpoint"
            )
            threshold = st.slider(
                "Confidence Threshold",
                min_value=0.5,
                max_value=1.0,
                value=0.80,
                step=0.05
            )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_dir = Path(__file__).parent / "temp"
            temp_dir.mkdir(exist_ok=True)
            temp_image_path = temp_dir / uploaded_file.name
            
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("▶️ Run Complete Pipeline", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    # Check if model exists
                    model_path_obj = Path(model_path)
                    if not model_path_obj.exists():
                        st.warning(f"Model not found at {model_path}. Running preprocessing only.")
                        model_path = None
                    
                    results = run_inference_pipeline(
                        str(temp_image_path),
                        model_path=model_path,
                        threshold=threshold
                    )
                    
                    if results and "error" not in results:
                        st.success("✓ Pipeline completed successfully!")
                        
                        # Display results
                        st.markdown("#### Outputs")
                        
                        # Show board and grid side by side
                        result_cols = st.columns(2)
                        
                        with result_cols[0]:
                            st.markdown("**Reconstructed Chessboard**")
                            if results["board_svg"]:
                                st.components.v1.html(results["board_svg"], height=512)
                            else:
                                st.info("Board visualization not available")
                        
                        with result_cols[1]:
                            st.markdown("**64 Classified Squares (8×8 Grid)**")
                            if results.get("grid") is not None:
                                st.image(cv2.cvtColor(results["grid"], cv2.COLOR_BGR2RGB), width='stretch')
                            else:
                                st.info("Grid visualization not available")
                        
                        st.markdown("---")
                        
                        # FEN output
                        if results["fen"]:
                            st.markdown("#### FEN Notation")
                            st.code(results["fen"], language="text")
                            
                            # Statistics
                            st.markdown("#### Classification Statistics")
                            stats_cols = st.columns(4)
                            
                            with stats_cols[0]:
                                st.metric("Total Squares", "64")
                            
                            with stats_cols[1]:
                                if results["labels"]:
                                    empty_count = results["labels"].count("empty")
                                    st.metric("Empty Squares", empty_count)
                            
                            with stats_cols[2]:
                                if results["labels"]:
                                    piece_count = sum(1 for l in results["labels"] if l not in ["empty", "unknown"])
                                    st.metric("Pieces Detected", piece_count)
                            
                            with stats_cols[3]:
                                if results["unknown_indices"]:
                                    st.metric("Unknown/Occluded", len(results["unknown_indices"]), delta_color="inverse")
                                else:
                                    st.metric("Unknown/Occluded", 0)
                            
                            # Confidence distribution
                            if results["confidences"]:
                                st.markdown("#### Confidence Distribution")
                                import pandas as pd
                                conf_data = pd.DataFrame({
                                    "Square": [f"{i:02d}" for i in range(64)],
                                    "Label": results["labels"],
                                    "Confidence": results["confidences"]
                                })
                                
                                # Show low confidence squares
                                low_conf = conf_data[conf_data["Confidence"] < threshold].sort_values("Confidence")
                                if not low_conf.empty:
                                    st.markdown("**Low Confidence Predictions (marked as unknown):**")
                                    st.dataframe(low_conf, width='content')
                                else:
                                    st.success("All predictions above threshold!")
                    
                    elif results and "error" in results:
                        st.error(f"Error: {results['error']}")
                    else:
                        st.error("Inference failed. Please check your model and image.")
    
    # Tab 4: Live Demo (Consolidated)
    with tab4:
        st.markdown('<div class="sub-header">Live Inference Demo</div>', 
                    unsafe_allow_html=True)
        
        st.markdown("""
        Upload your own chessboard image and run the complete pipeline step-by-step with a trained model.
        """)
        
        st.markdown("---")
        
        # Step 1: Image Upload
        st.markdown("### 📤 Step 1: Load Chessboard Image")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Upload option
            uploaded_file = st.file_uploader(
                "Upload a chessboard image",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a photo of a chessboard",
                key="live_demo_upload"
            )
            
            if uploaded_file is not None:
                # Convert uploaded file to numpy array
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                st.session_state.original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.success(f"✓ Uploaded: {uploaded_file.name}")
        
        with col2:
            if st.session_state.original_image is not None:
                # Convert BGR to RGB for display
                display_img = cv2.cvtColor(st.session_state.original_image, cv2.COLOR_BGR2RGB)
                st.image(display_img, caption="Original Image", width=300)
                
                # Image info
                h, w = st.session_state.original_image.shape[:2]
                st.info(f"📐 Image Size: {w} × {h} pixels")
            else:
                st.info("👆 Upload or select an image to begin")
        
        # Step 2: Preprocessing
        if st.session_state.original_image is not None:
            st.markdown("---")
            st.markdown("### 🔄 Step 2: Preprocessing Pipeline")
            
            if st.button("▶️ Run Preprocessing", type="primary", key="live_demo_preprocess"):
                with st.spinner("Detecting board and extracting squares..."):
                    warped_board, squares = preprocess_image(st.session_state.original_image)
                    
                    if warped_board is not None:
                        st.session_state.warped_board = warped_board
                        st.session_state.squares = squares
                        st.success("✓ Preprocessing complete!")
                    else:
                        st.error("❌ Board detection failed. Try another image.")
            
            if st.session_state.warped_board is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Warped Board (512×512)**")
                    display_warped = cv2.cvtColor(st.session_state.warped_board, cv2.COLOR_BGR2RGB)
                    st.image(display_warped, width=300)
                
                with col2:
                    st.markdown("**64 Extracted Squares**")
                    if st.session_state.squares is not None:
                        grid_img = create_grid_image(st.session_state.squares)
                        if grid_img is not None:
                            display_grid = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
                            st.image(display_grid, width=300)
                        
                        st.success(f"✓ Extracted {len(st.session_state.squares)} squares")
        
        # Step 3: Classification
        if st.session_state.squares is not None:
            st.markdown("---")
            st.markdown("### 🧠 Step 3: Piece Classification")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                model_path = st.text_input(
                    "Model Path",
                    value="model/resnet18_ft_blocks_black.pth",
                    help="Path to trained model checkpoint",
                    key="live_demo_model_path"
                )
            
            with col2:
                confidence_threshold = st.slider(
                    "Confidence Threshold (OOD)",
                    min_value=0.5,
                    max_value=1.0,
                    value=0.80,
                    step=0.05,
                    help="Predictions below this are marked as 'unknown'",
                    key="live_demo_threshold"
                )
            
            with col3:
                st.metric("Model", "ResNet18")
                st.metric("Classes", "13")
            
            if st.button("▶️ Run Classification", type="primary", use_container_width=True, key="live_demo_classify"):
                if not Path(model_path).exists():
                    st.error(f"❌ Model not found at {model_path}")
                else:
                    with st.spinner("Running inference..."):
                        # Save current image temporarily
                        temp_dir = Path(__file__).parent / "temp"
                        temp_dir.mkdir(exist_ok=True)
                        temp_image = temp_dir / "live_demo_image.jpg"
                        cv2.imwrite(str(temp_image), st.session_state.original_image)
                        
                        # Run inference
                        results = run_inference_pipeline(
                            str(temp_image),
                            model_path=model_path,
                            threshold=confidence_threshold
                        )
                        
                        if results and "error" not in results:
                            # Store results in session state
                            st.session_state.labels = results["labels"]
                            st.session_state.confidences = results["confidences"]
                            st.session_state.unknown_indices = results["unknown_indices"]
                            st.session_state.fen = results["fen"]
                            st.session_state.board_svg = results["board_svg"]
                            st.session_state.predictions = results.get("predictions", [])
                            
                            st.success("✓ Classification complete! See results below.")
                        elif results and "error" in results:
                            st.error(f"❌ Error: {results['error']}")
                        else:
                            st.error("❌ Inference failed. Please check your model and setup.")
        
        # Step 4: Results
        if st.session_state.labels and st.session_state.fen:
            st.markdown("---")
            st.markdown("### 📊 Step 4: Results & Board Reconstruction")
            
            # Statistics
            stats_cols = st.columns(4)
            
            with stats_cols[0]:
                st.metric("Total Squares", "64")
            
            with stats_cols[1]:
                empty_count = st.session_state.labels.count("empty")
                st.metric("Empty Squares", empty_count)
            
            with stats_cols[2]:
                piece_count = sum(1 for l in st.session_state.labels if l not in ["empty", "unknown"])
                st.metric("Pieces Detected", piece_count)
            
            with stats_cols[3]:
                unknown_count = len(st.session_state.unknown_indices)
                st.metric("Unknown/Occluded", unknown_count, delta_color="inverse")
            
            st.markdown("---")
            
            # Show board and grid side by side
            st.markdown("**Outputs:**")
            
            output_cols = st.columns(2)
            
            with output_cols[0]:
                st.markdown("Reconstructed Chessboard")
                if st.session_state.board_svg:
                    st.components.v1.html(st.session_state.board_svg, height=450)
            
            with output_cols[1]:
                st.markdown("64 Classified Squares (8×8 Grid)")
                # Try to load grid from inference results
                temp_output = Path(__file__).parent / "temp" / "inference_output"
                grid_path = temp_output / "crops_grid.jpg"
                if grid_path.exists():
                    grid_img = cv2.imread(str(grid_path))
                    if grid_img is not None:
                        st.image(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB), width='stretch')
                elif st.session_state.squares is not None:
                    # Fallback: generate grid from squares
                    grid_img = create_grid_image(st.session_state.squares, st.session_state.labels, st.session_state.predictions)
                    if grid_img is not None:
                        st.image(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB), width='stretch')
            
            st.markdown("---")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**FEN Notation**")
                st.code(st.session_state.fen, language="text")
            
            with col2:
                st.markdown("**Statistics & Confidence**")
                
                st.markdown("""
                **FEN Components:**
                - White: `P N B R Q K`
                - Black: `p n b r q k`
                - Empty: numbers
                - Unknown: `?`
                """)
                
                # Show low confidence predictions
                if st.session_state.confidences:
                    import pandas as pd
                    
                    threshold = confidence_threshold if 'confidence_threshold' in locals() else 0.80
                    low_conf_indices = [i for i, c in enumerate(st.session_state.confidences) if c < threshold]
                    
                    if low_conf_indices:
                        st.markdown("**⚠️ Low Confidence:**")
                        conf_data = pd.DataFrame({
                            "Sq": [f"{i:02d}" for i in low_conf_indices],
                            "Label": [st.session_state.labels[i] for i in low_conf_indices],
                            "Conf": [f"{st.session_state.confidences[i]:.2%}" for i in low_conf_indices]
                        })
                        st.dataframe(conf_data, width='content', hide_index=True)
                    else:
                        st.success("✓ All predictions confident!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>Chessboard Recognition System</strong></p>
        <p>Introduction to Deep Learning Course • Ben-Gurion University of the Negev • 2026</p>
        <p style="color: #888; font-size: 0.9rem;">Shon Grinberg • David Paster • Rotem Arie</p>
        <p>Technologies: PyTorch • OpenCV • Streamlit</p>
        <p style="margin-top: 1rem;">
            <a href="https://github.com/rotemarie/chessboard-recon" target="_blank" style="color: #2E86AB; text-decoration: none;">
                📁 View Project on GitHub
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

