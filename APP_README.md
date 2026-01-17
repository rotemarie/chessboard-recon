# 🎨 Chessboard Recognition Demo App

Interactive web-based UI for presenting the chess recognition project.

## 🚀 Quick Start

### 1. Install Streamlit (if not already installed)

```bash
# Activate virtual environment
source bguenv/bin/activate

# Install streamlit
pip install streamlit
```

### 2. Run the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## 📱 Features

### ✅ Currently Implemented

**Tab 1: Project Overview**
- Problem statement and key challenges
- Solution approach overview
- Dataset description
- Results summary

**Tab 2: Pipeline**
- Complete step-by-step walkthrough of the entire process
- Data collection and organization
- Preprocessing pipeline details
- Dataset preparation and splitting
- Model architecture and training
- OOD detection methodology
- Board reconstruction approach
- Perfect for class presentation!

**Tab 3: Demo - Input**
- Upload custom chessboard images (JPG, PNG)
- Select from sample images (game2 frames)
- Preview uploaded/selected image
- Display image dimensions

**Tab 4: Demo - Preprocessing**
- Run board detection and perspective transformation
- Display original vs warped board side-by-side
- Extract and visualize 64 individual squares
- Show 8×8 grid with square positions

**Tab 5: Demo - Classification** (Placeholder)
- Model configuration display
- OOD confidence threshold slider
- Placeholder for model inference
- Instructions for integration

**Tab 6: Demo - Results** (Placeholder)
- Placeholder for classification results
- Placeholder for FEN notation output
- Performance metrics display

### 🔧 To Be Integrated

1. **Model Loading**
   - Load trained ResNet18 checkpoint
   - Initialize model on CPU/GPU

2. **Inference Pipeline**
   - Preprocess squares (resize to 224×224, normalize)
   - Run model prediction on all 64 squares
   - Apply confidence thresholding for OOD detection

3. **Results Display**
   - Show predictions on 8×8 grid
   - Display confidence scores
   - Highlight occluded/unknown squares
   - Generate FEN notation with '?' for unknowns

4. **Visualizations**
   - Confusion matrix for predictions
   - Confidence distribution plot
   - Per-class accuracy breakdown

## 🎨 UI Structure

```
┌─────────────────────────────────────────┐
│  ♟️ Chessboard Recognition System       │
│  Deep Learning Project - BGU 2026       │
├─────────────────────────────────────────┤
│                                         │
│  📷 Input | 🔄 Preprocessing | 🤖 Cla…│
│                                         │
│  [Main Content Area]                    │
│                                         │
│  • Upload/select image                  │
│  • View preprocessing steps             │
│  • See classification results           │
│  • View FEN reconstruction              │
│                                         │
└─────────────────────────────────────────┘

Sidebar:
┌──────────────────┐
│ 🎯 Project Info  │
│ 📊 Statistics    │
│ 🏗️ Architecture  │
└──────────────────┘
```

## 🎬 Presentation Flow (For Class)

### Recommended Order:

**1. Start with "Project Overview" Tab (5 minutes)**
   - Explain the problem and challenges
   - Show the solution approach
   - Present dataset statistics
   - Highlight key results

**2. Go to "Pipeline" Tab (10-15 minutes)**
   - Walk through each section step-by-step:
     1. Data Collection
     2. Preprocessing (board detection, warping, extraction)
     3. Dataset Preparation (splitting, balancing)
     4. Model Training (architecture, hyperparameters)
     5. OOD Detection (confidence thresholding)
     6. Board Reconstruction
   - This is your main technical presentation!

**3. Switch to "Demo - Input" Tab (2 minutes)**
   - Select a sample image
   - Show the interface for loading data

**4. Go to "Demo - Preprocessing" Tab (3 minutes)**
   - Click "Run Preprocessing"
   - Show live board detection
   - Display warped board and 64 extracted squares
   - Emphasize real-time processing

**5. Show "Demo - Classification" Tab (1 minute)**
   - Explain model integration (placeholder)
   - Show OOD threshold slider
   - Mention future integration

**6. Show "Demo - Results" Tab (1 minute)**
   - Explain FEN output format
   - Show performance metrics
   - Discuss future improvements

**Total Time: ~20-25 minutes**

## 🔌 Integration Guide

To add model inference, modify `app.py`:

### Step 1: Load Model

```python
import torch
from training.model import load_model

@st.cache_resource
def load_chess_model():
    model = load_model(
        'checkpoints/best_model.pth',
        model_name='resnet18',
        device='cpu'
    )
    return model

model = load_chess_model()
```

### Step 2: Add Inference Function

```python
def classify_squares(squares, model, threshold=0.8):
    from torchvision import transforms
    from training.utils import IMAGENET_MEAN, IMAGENET_STD
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    predictions = []
    confidences = []
    
    for square in squares:
        square_rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        tensor = transform(square_rgb).unsqueeze(0)
        
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = probs.max(dim=1)
        
        pred_idx = pred.item()
        conf_val = conf.item()
        
        if conf_val < threshold:
            predictions.append('unknown')
        else:
            predictions.append(class_names[pred_idx])
        
        confidences.append(conf_val)
    
    return predictions, confidences
```

### Step 3: Update Classification Tab

Replace the placeholder in Tab 3 with:

```python
if st.button("🤖 Run Classification", type="primary"):
    predictions, confidences = classify_squares(
        st.session_state.squares, 
        model, 
        confidence_threshold
    )
    st.session_state.predictions = predictions
    st.session_state.confidences = confidences
    st.success("✅ Classification complete!")
```

### Step 4: Update Results Tab

Add visualization in Tab 4:

```python
if st.session_state.predictions is not None:
    # Display grid with predictions
    grid_img = create_grid_image(
        st.session_state.squares,
        predictions=st.session_state.predictions
    )
    st.image(grid_img)
    
    # Generate FEN
    fen = FENParser.labels_to_fen(st.session_state.predictions)
    st.code(fen, language='text')
```

## 📊 Customization

### Change Theme

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#2E86AB"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Add More Features

- **Export Results**: Add download button for predictions
- **Batch Processing**: Process multiple images
- **Comparison Mode**: Compare model predictions vs ground truth
- **Performance Analysis**: Show real-time accuracy metrics

## 🐛 Troubleshooting

### Port Already in Use

```bash
streamlit run app.py --server.port 8502
```

### Module Not Found

```bash
# Make sure virtual environment is activated
source bguenv/bin/activate

# Install missing packages
pip install -r requirements.txt
```

### Image Upload Issues

- Supported formats: JPG, JPEG, PNG
- Max file size: 200MB (Streamlit default)
- Try using sample images if upload fails

## 📝 Notes

- The app runs locally on your machine
- No data is sent to external servers
- Sample images are loaded from `data/per_frame/game2_per_frame/`
- Preprocessing uses the same pipeline as training

## 🎯 Presentation Tips

### Key Points to Emphasize:

1. **Problem Complexity**
   - Not just image classification - multi-step pipeline
   - Real-world challenges (angles, lighting, occlusions)
   - No temporal information (single frame only)

2. **Technical Decisions**
   - Why split by game, not frame (data leakage)
   - Why weighted sampling (class imbalance)
   - Why confidence thresholding (OOD detection)
   - Why ResNet18 over larger models (efficiency)

3. **Results**
   - 89% accuracy is good for real-world data
   - 92% preprocessing success rate
   - OOD detection works (85% TPR)

4. **Live Demo**
   - Show preprocessing in real-time
   - Emphasize the perspective transformation
   - Display the 8×8 grid clearly

### Questions You Might Get:

**Q: Why not use video/temporal information?**
A: Project requirement - single static images only. But temporal modeling is listed as future work.

**Q: Why only 89% accuracy?**
A: Real-world challenges - occlusions account for ~13% of errors, piece cropping ~27%. Clean images perform better.

**Q: How do you handle different board styles?**
A: Current model trained on specific dataset. Transfer learning or retraining would be needed for different boards.

**Q: What about occluded pieces?**
A: That's our OOD detection - confidence < 0.8 → marked as "unknown" in FEN output.

### Technical Details Ready:

- Dataset: 517 frames, ~30K squares
- Model: ResNet18, 11M params
- Training: SGD, early stopping, weighted sampling
- Preprocessing: OpenCV, edge detection, homography
- OOD: MSP (Maximum Softmax Probability)

---

**Ready for presentation!**

The app now has:
- ✅ Complete project documentation
- ✅ Step-by-step pipeline explanation
- ✅ Live preprocessing demo
- ✅ Professional appearance (fewer emojis)
- ✅ Clear structure for teaching

Add model integration when ready to make it fully functional.

