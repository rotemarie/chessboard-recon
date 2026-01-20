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


def run_inference_pipeline(image_path, model_path=None, threshold=0.50):
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Project Overview", "Work Process", "Pipeline", "Full Demo", "Live Demo"
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
    
    # Tab 2: Work Process (with nested tabs for challenges)
    with tab2:
        st.markdown('<div class="sub-header">Development Process & Challenges</div>', 
                    unsafe_allow_html=True)
        
        # Create nested tabs for challenges
        challenge_tab1, challenge_tab2, challenge_tab3, challenge_tab4 = st.tabs([
            "⚖️ Challenge 1: Imbalanced Dataset", 
            "🤖 Challenge 2: Model Selection", 
            "🔄 Challenge 3: Preprocessing Methods",
            "🎯 Challenge 4: OOD Detection"
        ])
        
        # Challenge 1: Imbalanced Dataset
        with challenge_tab1:
            st.markdown("### Challenge 1: Dealing with Imbalanced Dataset")
            
            st.markdown("---")
            
            # Problem Statement
            st.markdown("#### The Problem")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("""
                Our dataset exhibits severe class imbalance:
                
                - **Enormous amount of empty cell images** (~69% of dataset)
                - **Large amount of pawn images** (~16% of dataset)
                - **Tiny amount of queen images** (<2% of dataset)
                
                **Impact on Training:**
                - The loss function doesn't represent model performance
                - A naive "empty" predictor would reach **75% train accuracy** and **55% validation accuracy**
                - The model tends to **overpredict 'empty' and 'pawn'**
                - The model tends to **underpredict 'queen'** and other rare pieces
                """)
            
            with col2:
                st.warning("""
                **Why is this a problem?**
                
                Standard cross-entropy loss treats all classes equally. 
                With imbalanced data, the model learns to predict frequent classes 
                (empty, pawn) more often, achieving good loss but poor per-class accuracy.
                
                This is especially problematic for chess:
                - Missing a queen is much more critical than missing a pawn
                - Empty square classification is important but shouldn't dominate
                """)
            
            st.markdown("---")
            
            # Visualizations
            st.markdown("#### Dataset Distribution Analysis")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                dist_path = Path(__file__).parent / "for_ui" / "unbalanced" / "dist_train_vs_val.png"
                if dist_path.exists():
                    st.image(str(dist_path), 
                             caption="Distribution: Train vs Validation", 
                             width='stretch')
                else:
                    st.error("Image not found: dist_train_vs_val.png")
            
            with viz_col2:
                hist_path = Path(__file__).parent / "for_ui" / "unbalanced" / "hist vs val.png"
                if hist_path.exists():
                    st.image(str(hist_path), 
                             caption="Histogram: Train vs Validation", 
                             width='stretch')
                else:
                    st.error("Image not found: hist vs val.png")
            
            st.markdown("---")
            
            # Solutions
            st.markdown("#### Solutions We Explored")
            
            sol_col1, sol_col2, sol_col3 = st.columns(3)
            
            with sol_col1:
                st.markdown("##### 1️⃣ Undersampling")
                st.markdown("""
                **Approach:**
                Even out the number of samples for each class to match the minority class.
                
                **Pros:**
                - Simple to implement
                - Balanced training batches
                - No bias towards majority classes
                
                **Cons:**
                - ❌ **Expensive data lost** (throw away ~80% of data!)
                - Reduced effective dataset size
                - May underfit due to limited data
                """)
            
            with sol_col2:
                st.markdown("##### 2️⃣ Weighted Cross-Entropy Loss")
                st.markdown("""
                **Approach:**
                Assign higher weights to minority classes in the loss function:
                """)
                
                st.latex(r"\ell_n = -\sum_{c=1}^{C} w_c \log \frac{\exp(x_{n,c})}{\sum_{i=1}^{C} \exp(x_{n,i})}")
                
                st.markdown("""
                **Pros:**
                - Uses all available data
                - Mathematically principled
                
                **Cons:**
                - ❌ **Val set and training set differ in distribution**
                - Requires defining two different losses (train vs val)
                - Complicates training pipeline
                """)
            
            with sol_col3:
                st.markdown("##### 3️⃣ Weighted Sampler ✓")
                st.markdown("""
                **Approach:**
                Assign sampling probability to each image. We chose:
                
                **P(image) = 1 / (number of images in its class)**
                
                This ensures equal probability for each class to be drawn.
                
                **Pros:**
                - ✓ **Simple to implement**
                - ✓ **Utilizes all data available**
                - ✓ **No different loss functions needed**
                
                **Cons:**
                - Some images may never be seen in an epoch
                - Evaluation becomes not fully reproducible
                """)
            
            st.success("""
            **✓ Final Choice: Weighted Sampler**
            
            We chose weighted sampling because it's simple, utilizes all available data, 
            and doesn't require different loss functions for validation and test sets.
            """)
            
            st.markdown("---")
            
            # Results
            st.markdown("#### Impact of Weighted Sampling")
            
            st.markdown("""
            Below is a comparison of batch composition **before** and **after** applying weighted sampling:
            """)
            
            batch_path = Path(__file__).parent / "for_ui" / "chosen1" / "mean_batch_count.png"
            if batch_path.exists():
                st.image(str(batch_path), 
                         caption="Mean Batch Count Before and After Weighted Sampler", 
                         width=700)
            else:
                st.error("Image not found: mean_batch_count.png")
            
            st.markdown("""
            **Observations:**
            - Before: Batches heavily dominated by empty squares and pawns
            - After: All classes represented more equally in each batch
            - This leads to better per-class accuracy, especially for rare pieces (queens, kings, knights)
            """)
        
        # Challenge 2: Model Selection
        with challenge_tab2:
            st.markdown("### Challenge 2: Selecting the Right Model Architecture")
            
            st.markdown("---")
            
            st.markdown("#### Exploring Different Architectures")
            
            st.markdown("""
            After addressing the class imbalance issue, we needed to select an appropriate model architecture. 
            We experimented with several popular CNN architectures, testing both **transfer learning** 
            (freezing backbone, training only final layer) and **fine-tuning** (training all layers) approaches.
            """)
            
            st.info("""
            **Important Note:** The validation set at this point in time was "dirty" - meaning it consisted 
            of both occluded and clean images. However, the portion of occluded images in both training and 
            validation sets is minimal, so the decision on which model to use is still logical and valid.
            """)
            
            st.markdown("---")
            
            # Show models comparison
            st.markdown("#### Model Comparison Results")
            
            models_path = Path(__file__).parent / "for_ui" / "models.png"
            if models_path.exists():
                st.image(str(models_path), 
                         caption="Comparison of Different Model Architectures and Training Strategies", 
                         width=800)
            else:
                st.error("Image not found: models.png")
            
            st.markdown("---")
            
            # Analysis
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.markdown("#### Key Findings")
                st.markdown("""
                **Best Performance:**
                - **ResNet18 (fine-tuned):** 89.78% validation accuracy
                - 11.7 million parameters
                - Good convergence
                - Reasonable training time
                
                **Other Results:**
                - ResNet18 (transfer learning): 62.98%
                - ResNet50 (fine-tuned): 86.83%
                - VGG16 (transfer learning): 77.63%
                """)
            
            with col2:
                st.markdown("#### Analysis & Decision")
                st.markdown("""
                **Why Fine-tuning outperformed Transfer Learning?**
                
                Chess piece classification is quite different from ImageNet's general object recognition:
                - Pieces have unique shapes and textures
                - Background (chessboard) is domain-specific
                - Fine-grained classification needed (bishop vs pawn)
                
                Fine-tuning allows the model to adapt all layers to the chess domain, 
                while transfer learning only trains the final classifier.
                
                **Why ResNet18 over ResNet50?**
                - Better accuracy (89.78% vs 86.83%)
                - Fewer parameters (11.7M vs 25.6M)
                - Faster training and inference
                - Less prone to overfitting on our dataset size
                """)
            
            st.success("""
            **✓ Final Choice: ResNet18 (Fine-tuned)**
            
            Sweet 89% is a good result! But can we improve it further? 
            
            → This question led us to Challenge 3: investigating preprocessing methods...
            """)
        
        # Challenge 3: Preprocessing Methods (updated content)
        with challenge_tab3:
            st.markdown("### Challenge 3: Improving Preprocessing to Boost Accuracy")
            
            st.markdown("---")
            
            st.markdown("#### Initial Approach: Board Detection & Square Cropping")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("""
                **Our initial preprocessing pipeline:**
                
                1. **Board Detection** - Locate the chessboard in the image using edge detection
                2. **Perspective Warp** - Transform to perfect top-down 512×512 view
                3. **Square Extraction** - Crop the board into exactly 64 squares (64×64 pixels each)
                4. **Individual Tagging** - Each square labeled with its piece type from FEN
                
                This seemed straightforward, but we encountered a significant problem...
                """)
            
            with col2:
                # Show demo preprocessing image (the one used in the demo)
                demo_preprocess_path = Path(__file__).parent / "output" / "processed.jpeg"
                if demo_preprocess_path.exists():
                    st.image(str(demo_preprocess_path), 
                             caption="Initial Preprocessing: Board Detection & Square Extraction",
                             width=300)
            
            st.markdown("---")
            
            # The Problem
            st.markdown("#### The Problem: Tall Pieces Get Cut Off")
            
            st.markdown("""
            The angle at which images were taken and the exact square cropping **cut off "tall" pieces** 
            like queens, kings, and bishops. The model couldn't see the distinctive tops of these pieces!
            """)
            
            # Show confusion matrix for original method
            conf_col1, conf_col2 = st.columns([1, 1])
            
            with conf_col1:
                st.markdown("**Original Method Confusion Matrix:**")
                regular_conf_path = Path(__file__).parent / "for_ui" / "regular_confusion_mtx.png"
                if regular_conf_path.exists():
                    st.image(str(regular_conf_path), 
                             caption="Poor Performance on Queen, King, Bishop",
                             width='stretch')
                else:
                    st.error("Image not found: regular_confusion_mtx.png")
                
                st.warning("""
                **Worst Performances:**
                - Queen: 54% accuracy
                - King: Multiple confusions
                - Bishop: 83% accuracy
                
                These are exactly the tall pieces!
                """)
            
            with conf_col2:
                st.markdown("**What the Model Sees:**")
                error_examples_path = Path(__file__).parent / "for_ui" / "error examples.png"
                if error_examples_path.exists():
                    st.image(str(error_examples_path), 
                             caption="Error Examples: Pieces Cut Off at Top",
                             width='stretch')
                else:
                    st.error("Image not found: error examples.png")
                
                st.markdown("""
                **Example 1:** Black king predicted as black queen - the distinctive cross on top is cut off!
                
                **Example 2:** Black bishop predicted as black queen - only seeing the rounded base, not the pointed top.
                """)
            
            st.markdown("---")
            
            # Solutions Attempted
            st.markdown("#### Solutions Attempted")
            
            st.markdown("""
            **Approach:** Instead of cropping strictly the tile area, allow some **buffer space** 
            so more of the figure remains visible.
            
            **Danger:** This also introduces more of neighboring figures into the frame and may 
            cause the model to classify them instead of the target piece.
            
            We experimented with different crop strategies, checking model performance on each:
            """)
            
            # Show preprocessing comparison image
            preprocessing_img_path = Path(__file__).parent / "for_ui" / "preprocessing.png"
            if preprocessing_img_path.exists():
                st.image(str(preprocessing_img_path), 
                         caption="Comparison of Different Preprocessing Methods & Validation Accuracy", 
                         width=750)
            else:
                st.error("Image not found: preprocessing.png")
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("##### Method 1: Original (64×64)")
                st.markdown("""
                **Exact square extraction**
                - No padding
                - 64×64 pixels per square
                - Direct crop from warped board
                - **Validation Accuracy: 91.46%**
                
                **Problem:** 
                Pieces extending beyond square boundaries get cut off, especially tall pieces 
                like kings and queens. The model loses important visual information about piece tops.
                """)
            
            with col2:
                st.markdown("##### Method 2: Padded Squares")
                st.markdown("""
                **Single square with padding**
                - 64×64 base + padding (20-30%)
                - Black border padding at edges
                - Captures piece tops
                - **Validation Accuracy: 90.36% (30% padding)**
                
                **Advantage:** 
                Captures piece tops without excessive context. Good balance between 
                capturing full piece and maintaining focus on target square.
                
                **Issue:** 
                Still misses neighboring context that could help distinguish similar pieces.
                """)
            
            with col3:
                st.markdown("##### Method 3: 3×3 Blocks ✓")
                st.markdown("""
                **3×3 square neighborhood**
                - 192×192 pixels (3×3 squares)
                - Black/mirrored padding at board edges
                - Target square centered in block
                - **Best Validation Accuracy: 92.51%** ✓
                
                **Advantage:** 
                Provides rich spatial context. Each square "sees" its 8 neighbors, helping 
                the model distinguish similar pieces based on board position and surrounding pieces.
                """)
            
            st.success("""
            **✓ Winner: 3×3 Blocks with black padding**
            
            Achieved the best validation accuracy: **92.51%** (up from 91.46% with original cropping!)
            """)
            
            st.markdown("---")
            
            # Show 3x3 confusion matrix
            st.markdown("#### Results: 3×3 Block Confusion Matrix")
            
            result_col1, result_col2 = st.columns([3, 2])
            
            with result_col1:
                conf_33_path = Path(__file__).parent / "for_ui" / "33_confusion_mtx.png"
                if conf_33_path.exists():
                    st.image(str(conf_33_path), 
                             caption="3×3 Blocks Confusion Matrix - Significant Improvements",
                             width='stretch')
                else:
                    st.error("Image not found: 33_confusion_mtx.png")
            
            with result_col2:
                st.info("""
                **Why 3×3 Blocks Work Better:**
                
                ✓ **Captures full pieces** - including tops of tall pieces (queens, kings, bishops)
                
                ✓ **Provides spatial context** - model sees neighboring squares to better distinguish similar pieces
                
                ✓ **Understands position** - can use board location and surrounding pieces as additional features
                
                ✓ **Clean edge handling** - black padding at board edges avoids artifacts from mirroring/stretching
                """)
            
            st.markdown("---")
            
            # New Problem: Shading
            st.markdown("#### Remaining Challenge: Black Queen vs Black Bishop Confusion")
            
            st.markdown("""
            While 3×3 blocks solved the problem of cut-off pieces, a new issue emerged: 
            **black queens are now confused with black bishops**.
            """)
            
            # Show queen vs bishop confusion
            queen_bishop_col1, queen_bishop_col2 = st.columns([2, 3])
            
            with queen_bishop_col1:
                queen_bishop_path = Path(__file__).parent / "for_ui" / "queen_v_bishop.png"
                if queen_bishop_path.exists():
                    st.image(str(queen_bishop_path), 
                             caption="Black Queen Confused with Black Bishop",
                             width='stretch')
                else:
                    st.error("Image not found: queen_v_bishop.png")
            
            with queen_bishop_col2:
                st.markdown("""
                **The Black Queen is now fully visible** (no longer cut off), and the model no longer 
                confuses it with the black king. However, a new confusion emerged: 
                **black queen ↔ black bishop**.
                
                Why is this happening when white pieces don't have the same issue?
                """)
            
            st.markdown("---")
            
            # Explanation: Shading Differences
            st.markdown("#### The Root Cause: Shading and Contrast")
            
            shading_col1, shading_col2 = st.columns([3, 2])
            
            with shading_col1:
                confusing_pieces_path = Path(__file__).parent / "for_ui" / "confusing_pieces.png"
                if confusing_pieces_path.exists():
                    st.image(str(confusing_pieces_path), 
                             caption="Comparison: White vs Black Pieces - Shading Visibility",
                             width='stretch')
                else:
                    st.error("Image not found: confusing_pieces.png")
            
            with shading_col2:
                st.markdown("""
                **White Pieces:**
                - Clear shadows and shading visible
                - Strong shape cues from internal contours
                - Distinct edges and boundaries
                
                **Black Pieces:**
                - Shadows barely visible (dark on dark)
                - Light reflections/blinks replace shadows
                - Weaker contours and edges
                
                → Black pieces are **darker + lower-contrast**, 
                so the model loses critical shape cues.
                """)
            
            st.markdown("---")
            
            # Solution Attempt: Contrast Enhancement
            st.markdown("#### Attempted Solution: Contrast Enhancement")
            
            st.markdown("""
            We attempted to make the model pay more attention to **figure contours** and **light traces** 
            by increasing image contrast using several techniques:
            
            - **Gamma correction** - Brighten dark objects without blowing out highlights
            - **CLAHE (Contrast Limited Adaptive Histogram Equalization)** - Local contrast boost on luminance channel
            - **Unsharp masking** - Mild sharpening to make borders clearer
            """)
            
            # Show before/after examples
            enhance_col1, enhance_col2 = st.columns(2)
            
            with enhance_col1:
                st.markdown("**Black Pieces Enhancement:**")
                black_shading_path = Path(__file__).parent / "for_ui" / "black_shading.png"
                if black_shading_path.exists():
                    st.image(str(black_shading_path), 
                             caption="Before/After: Contrast Enhancement on Black Pieces",
                             width='stretch')
                else:
                    st.error("Image not found: black_shading.png")
            
            with enhance_col2:
                st.markdown("**White Pieces Enhancement:**")
                white_shading_path = Path(__file__).parent / "for_ui" / "white_shading.png"
                if white_shading_path.exists():
                    st.image(str(white_shading_path), 
                             caption="Before/After: Contrast Enhancement on White Pieces",
                             width='stretch')
                else:
                    st.error("Image not found: white_shading.png")
            
            st.markdown("---")
            
            # Results
            st.markdown("#### Results: Contrast Enhancement Did Not Help")
            
            result_enhance_col1, result_enhance_col2 = st.columns([3, 2])
            
            with result_enhance_col1:
                shading_res_path = Path(__file__).parent / "for_ui" / "shading_training_res.png"
                if shading_res_path.exists():
                    st.image(str(shading_res_path), 
                             caption="Training Results: Contrast Enhancement Actually Worsened Performance",
                             width='stretch')
                else:
                    st.error("Image not found: shading_training_res.png")
            
            with result_enhance_col2:
                st.warning("""
                **Outcome:**
                
                The contrast enhancement approach **did not improve results**, 
                and actually **slightly worsened** the validation accuracy.
                
                **Why?**
                - The bishop and queen are genuinely similar shapes
                - Visible contours play the main role in distinguishing them
                - Simply enhancing contrast doesn't add the missing shape information
                - The model needs to learn subtle differences (crown vs pointed top)
                """)
            
            st.markdown("---")
            
            # Future Work
            st.markdown("#### Open Problem & Possible Solutions")
            
            st.info("""
            **Resolving the black bishop ↔ black queen confusion remains an open challenge.**
            
            Possible solutions to explore in future work:
            
            1. **Triplet Loss Training** - Use triplet loss to explicitly drive queen and bishop 
               embeddings apart in the feature space
            
            2. **Increase Dataset Size** - More training examples may help the model learn to 
               recognize the crown (the main visual difference between queen and bishop)
            
            3. **Attention Mechanisms** - Add attention layers to force the model to focus on 
               the top of pieces where the distinctive features are located
            
            4. **Multi-Scale Features** - Combine features from multiple scales to capture 
               both fine details (crown) and overall shape
            """)
        
        # Challenge 4: Out-of-Distribution (OOD) Detection
        with challenge_tab4:
            st.markdown("### Challenge 4: Detecting Occlusions (Out-of-Distribution)")
            
            st.markdown("---")
            
            st.markdown("#### The Problem: Occlusions")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("""
                **Challenge:** The model should recognize when pieces are **occluded** (blocked by hands, 
                other objects, or poor lighting).
                
                **Issue:** Occlusions are **not labeled** in the training set!
                - We don't have ground truth for "this is occluded"
                - Can't train a separate "occluded" class
                - Model will try to classify occluded pieces and likely get them wrong
                """)
            
            with col2:
                st.warning("""
                **Why not just label them?**
                
                Occlusions are:
                - Unpredictable (hands, shadows, etc.)
                - Temporary (in video, they appear/disappear)
                - Expensive to manually annotate
                
                We need an **automatic** solution.
                """)
            
            st.markdown("---")
            
            # Solution
            st.markdown("#### Solution: Train on Clean, Detect via Confidence")
            
            st.markdown("""
            **Our Approach:**
            1. Train the model while **ignoring the existence of occluded images**
            2. Occluded images add noise to training, but they're a small fraction (~1-2%) so noise impact is minimal
            3. After training, use **confidence thresholding** to detect occlusions
            """)
            
            # Show clean vs occluded confidence distribution
            clean_vs_occ_path = Path(__file__).parent / "for_ui" / "clean_vs_occluded.png"
            if clean_vs_occ_path.exists():
                st.image(str(clean_vs_occ_path), 
                         caption="Confidence Distribution: Clean vs Occluded Images", 
                         width=600)
            else:
                st.error("Image not found: clean_vs_occluded.png")
            
            st.markdown("---")
            
            # Key Insight
            st.markdown("#### Key Insight: Confidence Separation")
            
            insight_col1, insight_col2 = st.columns([2, 3])
            
            with insight_col1:
                st.markdown("""
                **Observation:**
                
                After training, we noticed the model tends to assign labels to occluded images 
                with **much lower confidence** than clean images.
                
                **Solution:**
                - Use a **confidence threshold**
                - Predictions below threshold → relabel as "occluded" (unknown)
                - Clear separation means minimal impact on correct predictions
                """)
            
            with insight_col2:
                # Show OOD graph
                ood_graph_path = Path(__file__).parent / "for_ui" / "ood_graph.png"
                if ood_graph_path.exists():
                    st.image(str(ood_graph_path), 
                             caption="Cumulative Distribution: Clear Separation Between Clean and Occluded",
                             width='stretch')
                else:
                    st.error("Image not found: ood_graph.png")
            
            st.markdown("---")
            
            # Results
            st.markdown("#### Results on Validation Set")
            
            result_col1, result_col2 = st.columns([3, 2])
            
            with result_col1:
                ood_results_path = Path(__file__).parent / "for_ui" / "OOD.png"
                if ood_results_path.exists():
                    st.image(str(ood_results_path), 
                             caption="OOD Detection Performance Metrics",
                             width='stretch')
                else:
                    st.error("Image not found: OOD.png")
            
            with result_col2:
                st.success("""
                **Performance Summary:**
                
                ✓ **Clean images:** High confidence maintained
                
                ✓ **Occluded images:** Successfully detected with low confidence
                
                ✓ **Clear separation:** Minimal false positives (clean images marked as occluded)
                
                ✓ **Practical benefit:** System can flag uncertain predictions for manual review
                """)
                
                st.info("""
                **Threshold Selection:**
                
                We chose a threshold of **0.5** (50% confidence).
                
                **Reasoning:**
                We prefer some occluded images to be missed (which are pretty rare) rather 
                than the already poorly predicted queen class to be hurt even more.
                
                This conservative threshold ensures:
                - Most clean images pass (especially queens)
                - Very low confidence predictions flagged
                - Minimal impact on rare piece classes
                """)
            
            st.markdown("---")
            
            # Comparison: 3x3 vs Original
            st.markdown("#### 3×3 Blocks vs Original Cropping")
            
            st.markdown("""
            Comparing OOD detection performance between our original cropping method and the 3×3 blocks approach:
            """)
            
            comparison_path = Path(__file__).parent / "for_ui" / "33_vs_og_ood.png"
            if comparison_path.exists():
                st.image(str(comparison_path), 
                         caption="Comparison: 3×3 Blocks vs Original Cropping for OOD Detection", 
                         width=800)
            else:
                st.error("Image not found: 33_vs_og_ood.png")
    
    # Tab 3: Pipeline (functionality and integration)
    with tab3:
        st.markdown('<div class="sub-header">Complete Processing Pipeline</div>', 
                    unsafe_allow_html=True)
        
        # Create nested tabs for pipeline functionality
        pipeline_tab1, pipeline_tab2, pipeline_tab3 = st.tabs([
            "🔄 Preprocessing Pipeline", "🧠 Training & OOD Detection", "🔗 Integration & Deployment"
        ])
        
        # Pipeline Tab 1: Preprocessing Pipeline
        with pipeline_tab1:
            st.markdown("### Preprocessing Pipeline Implementation")
            
            st.markdown("---")
            
            st.markdown("#### Technical Implementation: 3×3 Block Extraction")
        
            impl_col1, impl_col2 = st.columns(2)
            
            with impl_col1:
                st.markdown("**Step-by-Step Process:**")
                st.markdown("""
                1. **Board Detection** - Locate the chessboard in the image using edge detection
                2. **Perspective Warp** - Transform to perfect top-down 512×512 view
                3. **Padding** - Add 64px black border on all sides (total: 640×640)
                4. **Block Extraction**
                   - For each of 64 squares:
                     - Extract 3×3 neighborhood (192×192 pixels)
                     - Center on target square
                     - Include 8 surrounding squares
                5. **Labeling**
                   - Parse FEN notation
                   - Assign labels (13 classes)
                   - Validate 64 total squares
                """)
            
            with impl_col2:
                st.markdown("**Key Benefits:**")
                st.markdown("""
                ✓ **Spatial Context:**
                - Model sees neighboring pieces
                - Better distinguishes similar pieces
                - Understands board position
                
                ✓ **Complete Piece Coverage:**
                - Captures tall pieces fully
                - No cropping at boundaries
                - Consistent input size
                
                ✓ **Edge Handling:**
                - Black padding at board edges
                - No distortion or artifacts
                - Natural appearance
                """)
                
                st.code("""
# Block extraction parameters
Board size: 512×512 pixels
Square size: 64×64 pixels
Padding: 64 pixels (black)
Block size: 192×192 pixels (3×3 squares)
Total blocks: 64 (one per square)
                """, language="python")
        
        # Pipeline Tab 2: Training & OOD
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
                
                **Threshold selection:** 0.50
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
        
        # Pipeline Tab 3: Integration
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
    --threshold 0.50 \\
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
    threshold=0.50,
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
    
    # Tab 4: Full Demo (Interactive Walkthrough)
    with tab4:
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
                - **Threshold:** 0.50 (confidence below → "unknown")
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
                value=0.50,
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
    
    # Tab 5: Live Demo (Consolidated)
    with tab5:
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
                    value=0.50,
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
                    
                    threshold = confidence_threshold if 'confidence_threshold' in locals() else 0.50
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

