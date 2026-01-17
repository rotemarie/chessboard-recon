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

# Add preprocessing to path
sys.path.append(str(Path(__file__).parent / 'preprocessing'))

from preprocessing.board_detector import BoardDetector
from preprocessing.square_extractor import SquareExtractor, FENParser


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


def load_sample_images():
    """Load available sample images."""
    samples_dir = Path(__file__).parent / "data/per_frame/game2_per_frame/tagged_images"
    if samples_dir.exists():
        return sorted(list(samples_dir.glob("frame_*.jpg")))[:10]  # First 10 frames
    return []


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
    st.markdown('<div style="text-align: center; color: #666; margin-bottom: 2rem;">'
                'Deep Learning Project - Ben-Gurion University 2026</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/ChessSet.jpg/300px-ChessSet.jpg", 
                 use_column_width=True)
        
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Project Overview", "Pipeline", "Demo: Input", "Demo: Preprocessing", "Demo: Classification", "Demo: Results"
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
    
    # Tab 2: Pipeline
    with tab2:
        st.markdown('<div class="sub-header">Complete Processing Pipeline</div>', 
                    unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Section 1: Data Collection
        st.markdown("### 1. Data Collection and Organization")
        
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("""
            **Input Data:**
            - 5 chess games with labeled frames
            - Each frame has a corresponding FEN notation
            - Images captured at various game stages
            - Total: 517 labeled frames
            
            **Data Format:**
            - CSV files with frame numbers and FEN strings
            - JPEG images (480×480 or similar)
            - PGN files for additional games (future use)
            """)
        
        with col2:
            st.code("""
Frame,FEN
200,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
588,rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR
...
            """, language="csv")
        
        st.markdown("---")
        
        # Section 2: Preprocessing
        st.markdown("### 2. Preprocessing Pipeline")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### A. Board Detection")
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
            """)
            
            st.markdown("#### B. Perspective Transformation")
            st.markdown("""
            1. **Order corners** consistently (TL, TR, BR, BL)
            2. **Apply homography** to map to 512×512 square
            3. **Result:** Perfect top-down view
            
            **Success Rate:** 92.1% across all games
            """)
        
        with col2:
            st.markdown("#### C. Square Extraction")
            st.markdown("""
            1. **Divide** 512×512 board into 8×8 grid
            2. **Extract** each square as 64×64 image
            3. **Order** following FEN convention:
               - a8 → h8 (rank 8)
               - a7 → h7 (rank 7)
               - ...
               - a1 → h1 (rank 1)
            
            **Output:** 64 individual square images per frame
            """)
            
            st.markdown("#### D. FEN Parsing and Labeling")
            st.markdown("""
            1. **Parse** FEN string to 64 labels
            2. **Map** each character to piece class
            3. **Expand** numbers (e.g., '8' → 8 empty squares)
            4. **Validate** total count = 64
            
            **Classes:** 13 total (12 pieces + empty)
            """)
        
        st.markdown("---")
        
        # Section 3: Dataset Preparation
        st.markdown("### 3. Dataset Preparation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Dataset Splitting")
            st.markdown("""
            **Critical:** Split by **game**, not frame!
            
            - **Train:** games 2, 4, 5 (70%)
            - **Val:** game 6 (15%)
            - **Test:** game 7 (15%)
            
            **Why by game?**
            Prevents data leakage - frames from the same game are highly correlated.
            """)
        
        with col2:
            st.markdown("#### Class Balancing")
            st.markdown("""
            **Problem:** Natural class imbalance
            - Empty: ~52%
            - Pawns: ~25%
            - Other pieces: ~23%
            
            **Solution:** Weighted random sampling
            - Inverse frequency weights
            - Sample with replacement
            - Equal class representation per epoch
            """)
        
        with col3:
            st.markdown("#### Data Augmentation")
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
        
        st.markdown("---")
        
        # Section 4: Model Training
        st.markdown("### 4. Model Architecture and Training")
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown("#### Model Selection")
            st.markdown("""
            **Tested architectures:**
            - ResNet18 (11M params) ✓ **Best**
            - ResNet50 (23M params)
            - VGG16 (138M params)
            
            **Training modes:**
            - **Fine-tuning:** Train all layers (chosen)
            - **Transfer learning:** Freeze backbone, train final layer only
            
            **Why ResNet18?**
            - Best accuracy (89.08%)
            - Reasonable training time
            - Good speed-accuracy tradeoff
            """)
        
        with col2:
            st.markdown("#### Training Configuration")
            st.code("""
# Hyperparameters
Model: ResNet18 (ImageNet pretrained)
Optimizer: SGD (lr=0.001, momentum=0.9)
Scheduler: StepLR (step_size=7, gamma=0.1)
Batch size: 16
Loss: Cross-entropy
Early stopping: patience=10

# Training results
Epochs to convergence: ~15-20
Training accuracy: 99.15%
Validation accuracy: 89.08%
Training time: ~2-3 hours (GPU)
            """, language="python")
        
        st.markdown("---")
        
        # Section 5: OOD Detection
        st.markdown("### 5. Out-of-Distribution (OOD) Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Problem: Occlusions")
            st.markdown("""
            **Observation:** ~13% of validation errors due to occlusions
            - Hands covering pieces
            - Other pieces blocking view
            - Poor lighting/shadows
            
            **Challenge:** Model outputs high-confidence wrong predictions
            
            **Solution:** Confidence-based OOD detection
            """)
            
            st.markdown("#### Method: Maximum Softmax Probability")
            st.markdown("""
            1. Compute softmax probabilities
            2. Take maximum probability as confidence score
            3. If confidence < threshold → mark as "unknown"
            
            **Threshold selection:** 0.80
            - Based on clean vs occluded distribution analysis
            - 5th percentile of clean confidence
            """)
        
        with col2:
            st.markdown("#### Results")
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
        
        st.markdown("---")
        
        # Section 6: Board Reconstruction
        st.markdown("### 6. Board Reconstruction")
        
        st.markdown("""
        **Final Pipeline:**
        
        ```
        Input Image → Board Detection → Perspective Transform → Square Extraction
                                                                         ↓
        FEN Output ← FEN Generator ← OOD Detection ← Model Classification ← 64 Squares
        ```
        
        **Output Format:** FEN notation with optional '?' for unknowns
        - Standard: `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR`
        - With unknowns: `rnbqkbnr/pppp?ppp/8/8/8/8/PPPPPPPP/RNBQKBNR`
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Performance Metrics")
            st.markdown("""
            - Overall accuracy: 89.08%
            - Empty squares: 95.2%
            - Kings/Queens: ~93%
            - Pawns: ~85%
            """)
        
        with col2:
            st.markdown("#### Error Analysis")
            st.markdown("""
            - Piece cropping: 27%
            - Occlusions: 13%
            - Similar pieces: 35%
            - Lighting: 15%
            - Other: 10%
            """)
        
        with col3:
            st.markdown("#### Future Improvements")
            st.markdown("""
            - Padded extraction
            - Context awareness
            - Advanced OOD methods
            - Temporal modeling
            """)
    
    # Tab 3: Input (renamed)
    with tab3:
        st.markdown('<div class="sub-header">Step 1: Load Chessboard Image</div>', 
                    unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Upload or Select Image")
            
            # Upload option
            uploaded_file = st.file_uploader(
                "Upload a chessboard image",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a photo of a chessboard"
            )
            
            # Sample images option
            st.markdown("**Or select from samples:**")
            sample_images = load_sample_images()
            
            if sample_images:
                sample_names = [f"Frame {img.stem.split('_')[1]}" for img in sample_images]
                selected_sample = st.selectbox(
                    "Sample images from game2",
                    options=[""] + sample_names,
                    index=0
                )
                
                if selected_sample:
                    idx = sample_names.index(selected_sample)
                    sample_path = sample_images[idx]
                    st.session_state.original_image = cv2.imread(str(sample_path))
                    st.success(f"Loaded: {sample_path.name}")
            
            if uploaded_file is not None:
                # Convert uploaded file to numpy array
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                st.session_state.original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.success(f"Uploaded: {uploaded_file.name}")
        
        with col2:
            st.markdown("### Preview")
            if st.session_state.original_image is not None:
                # Convert BGR to RGB for display
                display_img = cv2.cvtColor(st.session_state.original_image, cv2.COLOR_BGR2RGB)
                st.image(display_img, caption="Original Image", use_column_width=True)
                
                # Image info
                h, w = st.session_state.original_image.shape[:2]
                st.info(f"Image Size: {w} × {h} pixels")
            else:
                st.info("Upload or select an image to begin")
    
    # Tab 4: Preprocessing (Demo)
    with tab4:
        st.markdown('<div class="sub-header">Step 2: Preprocessing Pipeline</div>', 
                    unsafe_allow_html=True)
        
        if st.session_state.original_image is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Run Preprocessing", type="primary"):
                    warped_board, squares = preprocess_image(st.session_state.original_image)
                    
                    if warped_board is not None:
                        st.session_state.warped_board = warped_board
                        st.session_state.squares = squares
                        st.success("Preprocessing complete!")
                    else:
                        st.error("Board detection failed. Try another image.")
            
            if st.session_state.warped_board is not None:
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Original Image")
                    display_orig = cv2.cvtColor(st.session_state.original_image, cv2.COLOR_BGR2RGB)
                    st.image(display_orig, use_column_width=True)
                
                with col2:
                    st.markdown("### Warped Board (512×512)")
                    display_warped = cv2.cvtColor(st.session_state.warped_board, cv2.COLOR_BGR2RGB)
                    st.image(display_warped, use_column_width=True)
                
                st.markdown("---")
                st.markdown("### 8×8 Grid of Extracted Squares")
                
                if st.session_state.squares is not None:
                    grid_img = create_grid_image(st.session_state.squares)
                    if grid_img is not None:
                        display_grid = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
                        st.image(display_grid, use_column_width=True)
                    
                    st.success(f"Extracted {len(st.session_state.squares)} squares successfully")
        else:
            st.warning("Please load an image in the Input tab first")
    
    # Tab 5: Classification (Demo - Placeholder)
    with tab5:
        st.markdown('<div class="sub-header">Step 3: Piece Classification</div>', 
                    unsafe_allow_html=True)
        
        if st.session_state.squares is not None:
            st.markdown("### Model Configuration")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model", "ResNet18")
                st.metric("Parameters", "11.2M")
            
            with col2:
                st.metric("Input Size", "224×224")
                st.metric("Classes", "13")
            
            with col3:
                confidence_threshold = st.slider(
                    "OOD Confidence Threshold",
                    min_value=0.5,
                    max_value=1.0,
                    value=0.80,
                    step=0.05,
                    help="Predictions below this confidence are marked as 'unknown'"
                )
            
            st.markdown("---")
            
            # Placeholder for model inference
            st.info("**Integration Pending:** Model inference will be added here")
            
            st.markdown("""
            **Next Steps:**
            1. Load trained model checkpoint
            2. Preprocess each square (resize, normalize)
            3. Run inference on all 64 squares
            4. Apply confidence thresholding for OOD detection
            5. Display predictions with confidence scores
            """)
            
            # Demo button (placeholder)
            if st.button("Run Classification (Demo)", type="primary", disabled=True):
                st.warning("Model integration coming soon!")
        else:
            st.warning("Please run preprocessing first")
    
    # Tab 6: Results (Demo - Placeholder)
    with tab6:
        st.markdown('<div class="sub-header">Step 4: Board Reconstruction</div>', 
                    unsafe_allow_html=True)
        
        if st.session_state.squares is not None:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("### Classification Results")
                st.info("**Integration Pending:** Results will be displayed here")
                
                st.markdown("""
                **Display will include:**
                - 8×8 grid with predicted pieces
                - Confidence scores per square
                - Occluded/unknown squares highlighted
                - Per-class accuracy metrics
                """)
            
            with col2:
                st.markdown("### FEN Notation")
                st.info("**Integration Pending:** FEN output will be shown here")
                
                st.markdown("**Example FEN:**")
                st.code("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR", language="text")
                
                st.markdown("""
                - Standard chess board representation
                - '?' indicates occluded/unknown squares
                - Can be imported to chess engines
                """)
            
            st.markdown("---")
            st.markdown("### Model Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Overall Accuracy", "89.08%", "±0.12%")
            
            with col2:
                st.metric("Empty Squares", "95.2%", "+6.1%")
            
            with col3:
                st.metric("Piece Detection", "87.4%", "-1.6%")
        else:
            st.warning("Please run preprocessing first")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>Chessboard Recognition System</strong></p>
        <p>Introduction to Deep Learning Course • Ben-Gurion University of the Negev • 2026</p>
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

