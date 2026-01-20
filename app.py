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
from preprocessing.create_block_dataset import BlockSquareExtractor

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
    if 'fen_clean' not in st.session_state:
        st.session_state.fen_clean = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'class_names' not in st.session_state:
        st.session_state.class_names = None
    if 'board_svg' not in st.session_state:
        st.session_state.board_svg = None
    if 'board_svg_clean' not in st.session_state:
        st.session_state.board_svg_clean = None
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
            save_grid=True,
            save_clean_board=True
        )
        
        # Read the results
        original = cv2.imread(str(image_path))
        warped = cv2.imread(str(temp_output / "warped_board.jpg"))
        fen_path = temp_output / "fen.txt"
        fen_clean_path = temp_output / "fen_clean.txt"
        svg_path = temp_output / "board.svg"
        svg_clean_path = temp_output / "fen.svg"
        predictions_path = temp_output / "predictions.json"
        grid_path = temp_output / "crops_grid.jpg"
        
        fen = fen_path.read_text(encoding="utf-8") if fen_path.exists() else None
        fen_clean = fen_clean_path.read_text(encoding="utf-8") if fen_clean_path.exists() else None
        board_svg = svg_path.read_text(encoding="utf-8") if svg_path.exists() else None
        board_svg_clean = svg_clean_path.read_text(encoding="utf-8") if svg_clean_path.exists() else None
        
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
            "fen_clean": fen_clean,
            "board_svg": board_svg,
            "board_svg_clean": board_svg_clean,
            "predictions": predictions
        }
    
    except Exception as e:
        import traceback
        return {"error": f"{str(e)}\n\n{traceback.format_exc()}"}


def preprocess_image(image_array, show_debug=False):
    """Preprocess the chess board image using 3x3 block extraction."""
    # Initialize processors
    detector = BoardDetector(board_size=512)
    block_extractor = BlockSquareExtractor(
        board_size=512,
        border_mode="constant",
        border_color="black"
    )
    
    # Detect and warp board
    with st.spinner("Detecting chessboard..."):
        warped_board = detector.detect_board(image_array, debug=show_debug)
    
    if warped_board is None:
        return None, None
    
    # Extract 3x3 blocks
    with st.spinner("Extracting 64 blocks (3×3 context)..."):
        blocks = block_extractor.extract_blocks(warped_board)
    
    return warped_board, blocks


def create_grid_image(squares, labels=None, predictions=None):
    """Create an 8x8 grid visualization of squares/blocks."""
    if squares is None or len(squares) == 0:
        return None
    
    # Detect the actual size of the input squares/blocks
    input_size = squares[0].shape[0]  # Could be 64x64 or 192x192
    
    # Dynamically scale visualization
    if input_size > 100:
        # For large blocks (192x192), scale down to fit in grid
        display_size = 80
    else:
        # For small squares (64x64), keep original size
        display_size = 64
    
    border = 2
    label_height = 30
    cell_size = display_size + border * 2
    
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
        
        # Resize square to display size if needed
        if square.shape[0] != display_size or square.shape[1] != display_size:
            resized_square = cv2.resize(square, (display_size, display_size))
        else:
            resized_square = square
        
        # Place square
        grid[y_start:y_start+display_size, x_start:x_start+display_size] = resized_square
        
        # Add label below
        file_letter = chr(ord('a') + col)
        rank_number = 8 - row
        position = f"{file_letter}{rank_number}"
        
        label_y = y_start + display_size + 20
        label_x = x_start + 10
        
        # Draw background
        cv2.rectangle(grid, 
                     (x_start, y_start + display_size),
                     (x_start + display_size, y_start + display_size + label_height),
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
        3. **Extract** 64 3×3 block crops (centered on each square)
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
            
            st.markdown("## System Architecture")
            
            architecture_path = Path(__file__).parent / "architecture.jpeg"
            if architecture_path.exists():
                st.image(str(architecture_path), 
                         caption="Complete Pipeline Architecture", 
                         width=700)
            else:
                st.error("Image not found: architecture.jpeg")
            
            # Add metrics below architecture
            st.markdown("### Overall Metrics")
            
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.markdown("""
                **Performance:**
                - Validation Accuracy: **92.51%**
                - Empty squares: **95.2%**
                - Kings/Queens: **~93%**
                - Pawns: **~87%**
                - Board detection: **92.1%**
                """)
            
            with metrics_col2:
                st.markdown("""
                **Error Analysis:**
                - Similar pieces: **35%**
                - Piece cropping: **27%**
                - Lighting: **15%**
                - Occlusions: **13%**
                - Other: **10%**
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
            
            # Add batch distribution image
            st.markdown("#### Batch Distribution (Imbalanced Dataset)")
            st.markdown("""
            **Problem:** Imbalanced dataset batches are dominated by majority class, 
            which causes only the loss on majority class to be optimized.
            """)
            
            batch_counts_path = Path(__file__).parent / "for_ui" / "counts_mean_batches2.jpeg"
            if batch_counts_path.exists():
                st.image(str(batch_counts_path), 
                         caption="Mean Counts of Batches - Dominated by Empty Squares", 
                         width=700)
            else:
                st.error("Image not found: counts_mean_batches2.jpeg")
            
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
            
            st.markdown("---")
            
            # Balanced vs Imbalanced Comparison
            st.markdown("#### Balanced Dataset vs Imbalanced Dataset")
            
            st.markdown("""
            To validate our approach, we compared training with a balanced dataset versus the original imbalanced dataset:
            
            - **Balanced model:** Trained with balanced training set and hyperparameters tuned on balanced validation set
            - **Imbalanced model:** Trained with imbalanced training set and hyperparameters tuned on imbalanced validation set
            
            Both models used the original tile cropping method (64×64 squares).
            """)
            
            balanced_path = Path(__file__).parent / "for_ui" / "balanced_v_unbalanced.png"
            if balanced_path.exists():
                st.image(str(balanced_path), 
                         caption="Comparison: Balanced vs Imbalanced Training", 
                         width=700)
            else:
                st.error("Image not found: balanced_v_unbalanced.png")
            
            st.info("""
            **Key Findings:**
            - Balanced model shows better validation accuracy
            - However, on a balanced clean test set, the balanced model performs better
            - On an imbalanced clean test set (closer to real-world distribution), both perform similarly
            - This validates our weighted sampler approach for handling class imbalance
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
                             width=400)
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
                             width=500)
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
                             width=500)
                else:
                    st.error("Image not found: black_shading.png")
            
            with enhance_col2:
                st.markdown("**White Pieces Enhancement:**")
                white_shading_path = Path(__file__).parent / "for_ui" / "white_shading.png"
                if white_shading_path.exists():
                    st.image(str(white_shading_path), 
                             caption="Before/After: Contrast Enhancement on White Pieces",
                             width=500)
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
                             width=600)
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
            st.markdown("#### Solution: Train on Dirty Data, Detect via Confidence")
            
            st.markdown("""
            **Our Approach:**
            1. Train the model on **dirty data** (includes occluded images)
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
        st.markdown('<div class="sub-header">Final Pipeline Architecture</div>', 
                    unsafe_allow_html=True)
        
        st.markdown("""
        This section describes our final implementation choices and system performance.
        """)
        
        st.markdown("---")
        
        # Section 1: Preprocessing
        st.markdown("### 1. Preprocessing: 3×3 Block Extraction")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Process:**")
            st.markdown("""
            1. **Board Detection** - Edge detection & perspective warp
            2. **Transform** - 512×512 top-down view
            3. **Padding** - Add 64px black border (640×640 total)
            4. **Block Extraction** - Extract 3×3 neighborhoods (192×192 each)
            5. **Labeling** - Parse FEN notation → 13 classes
            
            **Output:** 64 blocks, one per square
            """)
            
            st.code("""
# Final parameters
Board size: 512×512 pixels
Padding: 64 pixels (black)
Block size: 192×192 pixels
Total blocks: 64
            """, language="python")
        
        with col2:
            st.markdown("**Why 3×3 Blocks?**")
            st.markdown("""
            ✓ **Spatial Context** - Model sees neighboring pieces
            
            ✓ **Complete Piece Coverage** - Captures tall pieces fully
            
            ✓ **Better Distinction** - Helps differentiate similar pieces
            
            ✓ **Best Validation Accuracy** - 92.51% (vs 89% for single squares)
            
            **Comparison:**
            - Original crop: 89.08%
            - Padded (30%): 90.36%
            - **3×3 blocks: 92.51%** ✓
            """)
        
        st.markdown("---")
        
        # Section 2: Model Training
        st.markdown("### 2. Model Architecture & Training")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Model: ResNet18 (Fine-tuned)**")
            st.markdown("""
            **Architecture:**
            - Base: ResNet18 pre-trained on ImageNet
            - Input: 192×192 RGB images
            - Output: 13 classes (6 white + 6 black + empty)
            - Parameters: 11M
            
            **Why ResNet18?**
            - Best accuracy with 3×3 blocks
            - Reasonable training time (~2-3 hours)
            - Good speed-accuracy tradeoff
            """)
            
            st.code("""
# Training configuration
Optimizer: SGD (lr=0.001, momentum=0.9)
Scheduler: StepLR (step_size=7, gamma=0.1)
Batch size: 16
Loss: Cross-entropy
Class balancing: Weighted sampler
Early stopping: patience=10
            """, language="python")
        
        with col2:
            st.markdown("**Training Results:**")
            st.markdown("""
            **Dataset:**
            - 5 labeled games (517 frames)
            - ~30K labeled squares
            - Split by game (70/15/15)
            
            **Performance:**
            - Training accuracy: 99.15%
            - **Validation accuracy: 92.51%**
            - Epochs to convergence: ~15-20
            
            **Per-Class Performance:**
            - Empty squares: 95.2%
            - Kings/Queens: ~93%
            - Pawns: ~87%
            - Board detection: 92.1%
            """)
        
        st.markdown("---")
        
        # Section 3: OOD Detection
        st.markdown("### 3. Out-of-Distribution Detection")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Method: Confidence Thresholding**")
            st.markdown("""
            **Problem:** ~13% of errors due to occlusions (hands, shadows, etc.)
            
            **Solution:** Maximum Softmax Probability
            1. Compute softmax probabilities
            2. Take maximum as confidence score
            3. If confidence < 0.50 → mark as "unknown"
            
            **Threshold: 0.50**
            - Balances false positives vs true positives
            - Prefers missing some occlusions over mis-labeling clean images
            """)
        
        with col2:
            st.markdown("**Results:**")
            st.markdown("""
            **Clean images:**
            - Mean confidence: 0.94 ± 0.08
            - False positive rate: 4.8%
            
            **Occluded images:**
            - Mean confidence: 0.62 ± 0.21
            - True positive rate: 85.4%
            
            **Confidence separation: 0.32** → Clear distinction!
            
            **Output:** FEN with '?' for uncertain squares
            - Example: `rnbqk?nr/pppp1ppp/8/8/...`
            """)
        
        st.markdown("---")
        
        # Section 4: System Performance
        st.markdown("### 4. Overall System Performance")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.markdown("**Metrics**")
            st.metric("Validation Accuracy", "92.51%")
            st.metric("Board Detection", "92.1%")
            st.metric("OOD Detection", "85.4%")
        
        with metric_col2:
            st.markdown("**Error Sources**")
            st.markdown("""
            - Similar pieces: 35%
            - Piece cropping: 27%
            - Lighting: 15%
            - Occlusions: 13%
            - Other: 10%
            """)
        
        with metric_col3:
            st.markdown("**Output**")
            st.markdown("""
            - FEN notation
            - Board SVG
            - Confidence scores
            - 64 classified blocks
            - Unknown markers
            """)
        
        st.markdown("---")
        
        # Section 5: Usage
        st.markdown("### 5. How to Use")
        
        usage_col1, usage_col2 = st.columns(2)
        
        with usage_col1:
            st.markdown("**Command Line:**")
            st.code("""
python inference/pipeline.py \\
    --image board.jpg \\
    --model model/resnet18_ft_blocks_black.pth \\
    --threshold 0.80 \\
    --save-grid
            """, language="bash")
        
        with usage_col2:
            st.markdown("**Python API:**")
            st.code("""
from inference.pipeline import run_pipeline

run_pipeline(
    image_path="board.jpg",
    model_path="model/resnet18_ft_blocks_black.pth",
    threshold=0.80,
    save_grid=True
)
            """, language="python")
    
    # Tab 4: Full Demo (Interactive Walkthrough)
    with tab4:
        st.markdown('<div class="sub-header">Complete Pipeline Walkthrough</div>', 
                    unsafe_allow_html=True)
        
        st.markdown("""
        Click on each step below to see the pipeline in action.
        """)
        
        full_demo_dir = Path(__file__).parent / "full_demo"
        
        # Step 1: Input
        with st.expander("**📥 Step 1: Input Image**", expanded=True):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                og_path = full_demo_dir / "og.jpeg"
                if og_path.exists():
                    st.image(str(og_path), caption="Original Image", width=400)
                else:
                    st.error("Image not found: og.jpeg")
            
            with col2:
                st.markdown("""
                **Input:** Raw photo of a physical chessboard.
                
                **Challenges:**
                - Perspective distortion
                - Varying lighting
                - Background clutter
                - Different camera angles
                """)
        
        # Step 2: Preprocessing
        with st.expander("**🔄 Step 2: Preprocessing Pipeline**"):
            preprocessing_path = full_demo_dir / "preprocessing.jpeg"
            if preprocessing_path.exists():
                st.image(str(preprocessing_path), caption="64 Extracted 3×3 Blocks", width=700)
            else:
                st.error("Image not found: preprocessing.jpeg")
            
            st.markdown("""
            **Process:**
            1. **Board Detection** → Find chessboard corners
            2. **Perspective Warp** → Transform to 512×512 top-down view
            3. **3×3 Block Extraction** → Extract 64 blocks (192×192 pixels each)
            
            **Output:** 64 blocks ready for classification
            """)
        
        # Step 3: Classification & Output (combined)
        with st.expander("**🧠 Step 3: Model Classification & FEN Output**"):
            st.markdown("""
            **ResNet18 Classification:**
            - 13 classes (12 pieces + empty)
            - 92.5% validation accuracy
            
            **OOD Detection:**
            - Confidence threshold: 0.50
            - Low confidence → "?" (unknown)
            - Red X marks uncertain squares
            """)
            
            st.markdown("---")
            st.markdown("#### Model Outputs:")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**With OOD Detection (Uncertain squares marked):**")
                fen_dirty_path = full_demo_dir / "fen_dirty.jpeg"
                if fen_dirty_path.exists():
                    st.image(str(fen_dirty_path), caption="Red X = Low Confidence", width=400)
                else:
                    st.error("Image not found: fen_dirty.jpeg")
                
                # Show FEN with unknowns
                fen_dirty_txt = full_demo_dir / "fen.txt"
                if fen_dirty_txt.exists():
                    fen_dirty = fen_dirty_txt.read_text().strip()
                    st.code(fen_dirty, language="text")
                    st.caption("FEN with '?' for uncertain predictions")
            
            with col2:
                st.markdown("**Clean Output (Final Board):**")
                fen_clean_path = full_demo_dir / "fen_clean.jpeg"
                if fen_clean_path.exists():
                    st.image(str(fen_clean_path), caption="Final Reconstruction", width=400)
                else:
                    st.error("Image not found: fen_clean.jpeg")
                
                # Show clean FEN
                fen_clean_txt = full_demo_dir / "fen_clean.txt"
                if fen_clean_txt.exists():
                    fen_clean = fen_clean_txt.read_text().strip()
                    st.code(fen_clean, language="text")
                    st.caption("Standard FEN notation")
            
        
    
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
                st.info("👈 Upload or select an image to begin")
        
        # Step 2: Preprocessing
        if st.session_state.original_image is not None:
            st.markdown("---")
            st.markdown("### 🔄 Step 2: Preprocessing Pipeline")
            
            if st.button("▶️ Run Preprocessing", type="primary", key="live_demo_preprocess"):
                with st.spinner("Detecting board and extracting 3×3 blocks..."):
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
                    st.markdown("**64 Extracted 3×3 Blocks (192×192 each)**")
                    if st.session_state.squares is not None:
                        grid_img = create_grid_image(st.session_state.squares)
                        if grid_img is not None:
                            display_grid = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
                            st.image(display_grid, width=300)
                        
                        st.success(f"✓ Extracted {len(st.session_state.squares)} blocks (3×3 context per square)")
        
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
                            st.session_state.fen_clean = results.get("fen_clean")
                            st.session_state.board_svg = results["board_svg"]
                            st.session_state.board_svg_clean = results.get("board_svg_clean")
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
            
            # Show both versions of the board
            board_cols = st.columns(2)
            
            with board_cols[0]:
                st.markdown("**With OOD Detection** (Red X = Uncertain)")
                if st.session_state.board_svg:
                    st.components.v1.html(st.session_state.board_svg, height=550, scrolling=False)
                st.markdown("**FEN Notation (with OOD)**")
                st.code(st.session_state.fen, language="text")
                st.caption("'?' indicates low-confidence predictions")
            
            with board_cols[1]:
                st.markdown("**Clean Board** (Final Output)")
                if st.session_state.board_svg_clean:
                    st.components.v1.html(st.session_state.board_svg_clean, height=550, scrolling=False)
                elif st.session_state.board_svg:
                    st.components.v1.html(st.session_state.board_svg, height=550, scrolling=False)
                st.markdown("**FEN Notation (Clean)**")
                if st.session_state.fen_clean:
                    st.code(st.session_state.fen_clean, language="text")
                    st.caption("Ready for chess engines")
                else:
                    st.code(st.session_state.fen, language="text")
            
            st.markdown("---")
            
            # Show grid (smaller)
            st.markdown("**64 Classified Blocks (8×8 Grid):**")
            # Try to load grid from inference results
            temp_output = Path(__file__).parent / "temp" / "inference_output"
            grid_path = temp_output / "crops_grid.jpg"
            if grid_path.exists():
                grid_img = cv2.imread(str(grid_path))
                if grid_img is not None:
                    # Display grid at 60% width for smaller size
                    st.image(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB), width=800)
            elif st.session_state.squares is not None:
                # Fallback: generate grid from squares
                grid_img = create_grid_image(st.session_state.squares, st.session_state.labels, st.session_state.predictions)
                if grid_img is not None:
                    st.image(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB), width=800)
            
            st.markdown("---")
        
    
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

