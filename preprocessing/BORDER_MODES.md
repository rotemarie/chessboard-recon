# Border Handling for Dataset Creation

When creating padded or block-based datasets, we need to pad beyond the original board boundaries for edge squares. This document explains the different padding strategies.

## Applies To

- **Padded Dataset (`create_padded_dataset.py`)**: Single squares with 30% padding
- **Block Dataset (`create_block_dataset.py`)**: 3×3 blocks of squares

## The Problem

Edge squares don't have 8 neighbors - corner squares only have 3 neighbors, edge squares have 5. We need to "fill in" the missing neighbors somehow.

## Available Border Modes

### 1. **REFLECT (Default)** ✅ Recommended

- **What it does**: Mirrors the image at the border
- **Example**: `abc|dcb` (the `|` is the edge, `d` is mirrored from `d`)
- **Visual**: Clean, natural-looking, no artifacts
- **Pros**:
  - No blur or stretched pixels
  - Maintains image continuity
  - Standard approach in computer vision
- **Cons**: None for this use case
- **Usage**: `--border-mode reflect`

```
Original board edge:    Reflected padding:
┌─────┐                ┌─────┬─────┐
│ ♗ ♔ │                │ ♔ ♗ │ ♗ ♔ │
│ ♟ ♟ │       →        │ ♟ ♟ │ ♟ ♟ │
└─────┘                └─────┴─────┘
                       (mirrored)
```

### 2. **REPLICATE** ❌ Not Recommended (Creates Blur)

- **What it does**: Repeats the edge pixels
- **Example**: `abc|ccc` (the edge pixel `c` is repeated)
- **Visual**: Creates blur/smear at edges
- **Pros**: Simple
- **Cons**: 
  - **Creates blur artifacts** that confuse the model
  - Stretches edge pixels unnaturally
  - Model may learn to ignore edges
- **Usage**: `--border-mode replicate`

```
Original board edge:    Replicated padding:
┌─────┐                ┌─────┬─────┐
│ ♗ ♔ │                │ ♗ ♔ │ ♔ ♔ │ ← stretched
│ ♟ ♟ │       →        │ ♟ ♟ │ ♟ ♟ │ ← stretched
└─────┘                └─────┴─────┘
```

### 3. **CONSTANT** (Gray Fill)

- **What it does**: Fills with a constant color (gray: 128,128,128)
- **Example**: `abc|000` (filled with zeros/constant)
- **Visual**: Gray border around edge squares
- **Pros**:
  - Very clear boundary
  - Model knows this is "not a square"
  - No artificial patterns
- **Cons**:
  - Less natural
  - Model needs to learn to ignore gray areas
- **Usage**: `--border-mode constant`

```
Original board edge:    Constant padding:
┌─────┐                ┌─────┬─────┐
│ ♗ ♔ │                │ ♗ ♔ │ ███ │ ← gray
│ ♟ ♟ │       →        │ ♟ ♟ │ ███ │ ← gray
└─────┘                └─────┴─────┘
```

## Recommendation

**Use `reflect` (the default)**. It provides the most natural-looking images without artifacts and is the standard approach in image processing.

## Example Usage

```bash
# Default (reflect)
python create_block_dataset.py

# Explicit reflect
python create_block_dataset.py --border-mode reflect

# Try constant (gray fill)
python create_block_dataset.py --border-mode constant --output-root ../preprocessed_data_blocks_constant

# Don't use replicate (but if you must...)
python create_block_dataset.py --border-mode replicate
```

## Impact on Model Training

The border mode affects:
1. **Edge square classification accuracy** - Clean edges → better accuracy
2. **Feature learning** - Blur artifacts → model may learn wrong features
3. **Generalization** - Natural images → better real-world performance
4. **Training time** - Confusing artifacts → slower convergence

**Bottom line**: The `reflect` mode helps the model focus on actual chess pieces and patterns, not border artifacts.

