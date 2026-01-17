"""
Script to extract all plots from the Chess Piece Recognition notebook.
This script regenerates all plots and saves them as PNG files in the plots/ directory.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
from PIL import Image
import random

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Create plots directory
PLOTS_DIR = project_root / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

print(f"📊 Extracting plots to: {PLOTS_DIR}")
print("="*60)


def get_dataset_histogram(dataset):
    """Get class distribution histogram from dataset."""
    idx_to_class = {idx: cls for cls, idx in dataset.class_to_idx.items()}
    counts = Counter(dataset.targets)
    histogram = {idx_to_class[idx]: count for idx, count in counts.items()}
    return histogram


def to_dist(dataset_histo):
    """Convert histogram to distribution (frequencies)."""
    dataset_size = sum(dataset_histo.values())
    return {cls: count / dataset_size for cls, count in dataset_histo.items()}


def plot_dataset_histo(train_histo, val_histo, as_dist=False, y_lim=None, save_name=None):
    """Plot training vs validation data distribution."""
    piece_types = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']

    if as_dist:
        train_histo = to_dist(train_histo)
        val_histo = to_dist(val_histo)

    white_counts_train = [train_histo.get(f'white_{p}') for p in piece_types]
    black_counts_train = [train_histo.get(f'black_{p}') for p in piece_types]
    empty_count_train = train_histo.get('empty')

    white_counts_val = [val_histo.get(f'white_{p}') for p in piece_types]
    black_counts_val = [val_histo.get(f'black_{p}') for p in piece_types]
    empty_count_val = val_histo.get('empty')

    # Set up x-axis indices
    x = np.arange(len(piece_types))
    width = 0.2  # Width of each grouped bar

    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw grouped bars for pieces
    ax.bar(x - 1.5*width, white_counts_train, width, label='Train White', color='#f0d9b5', edgecolor='gray')
    ax.bar(x - 0.5*width, black_counts_train, width, label='Train Black', color='#b58863', edgecolor='gray')
    ax.bar(x + 0.5*width, white_counts_val, width, label='Val White', color='#d5e1df', edgecolor='blue', alpha=0.7)
    ax.bar(x + 1.5*width, black_counts_val, width, label='Val Black', color='#635a52', edgecolor='blue', alpha=0.7)

    # Draw single bar for the 'Empty' class at the end
    empty_pos = len(piece_types)
    ax.bar(empty_pos - 0.5*width, empty_count_train, width*2, label='Train Empty', color='#769656', edgecolor='gray')
    ax.bar(empty_pos + 0.5*width, empty_count_val, width*2, label='Val Empty', color='#4a6136', edgecolor='blue', alpha=0.7)

    # Formatting the graph
    if as_dist:
        ax.set_title('Distribution of Train vs Validation', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency')
    else:
        ax.set_title('Histogram of Train vs Validation', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count')

    # Set x-ticks to match piece types + the empty cell
    ax.set_xticks(list(x) + [empty_pos])
    ax.set_xticklabels([p.capitalize() for p in piece_types] + ['Empty'])

    if as_dist:
        ax.set_ylim(0, 1)

    if y_lim:
        ax.set_ylim(y_lim)
        
    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_name:
        save_path = PLOTS_DIR / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_name}")
    
    plt.close()


def plot_histo(histo, title, as_dist=False, y_lim=None, save_name=None):
    """Plot single histogram for a dataset or batch."""
    piece_types = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']

    if as_dist:
        total = sum(histo.values())
        histo = {k: v / total for k, v in histo.items()}

    white_counts = [histo.get(f'white_{p}', 0) for p in piece_types]
    black_counts = [histo.get(f'black_{p}', 0) for p in piece_types]
    empty_count = histo.get('empty', 0)

    x = np.arange(len(piece_types))
    width = 0.3

    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw grouped bars
    ax.bar(x - width/2, white_counts, width, label='White', color='#f0d9b5', edgecolor='gray')
    ax.bar(x + width/2, black_counts, width, label='Black', color='#b58863', edgecolor='gray')

    # Draw single bar for 'Empty'
    empty_pos = len(piece_types)
    ax.bar(empty_pos, empty_count, width*1.5, label='Empty', color='#769656', edgecolor='gray')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency' if as_dist else 'Count')
    ax.set_xticks(list(x) + [empty_pos])
    ax.set_xticklabels([p.capitalize() for p in piece_types] + ['Empty'])

    if y_lim:
        ax.set_ylim(y_lim)
    elif as_dist:
        ax.set_ylim(0, 1)

    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_name:
        save_path = PLOTS_DIR / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_name}")
    
    plt.close()


def tensor_to_numpy_image(image, imagenet_mean, imagenet_std):
    """Transform tensor to displayable numpy image."""
    image = image.numpy().transpose((1, 2, 0))
    image = imagenet_std * image + imagenet_mean
    image = image.clip(0, 1)
    return image


def plot_dataset_sample(N, dataset, dataset_name, imagenet_mean, imagenet_std, save_name=None):
    """Plot N random samples from each class."""
    num_classes = len(dataset.classes)
    fig, axes = plt.subplots(
        num_classes, N,
        figsize=(N * 2, num_classes * 2)
    )
    fig.suptitle('Random Samples from Each Class', y=1.0, fontsize=25)

    for row, cls in enumerate(dataset.classes):
        cls_idx = dataset.class_to_idx[cls]
        candidates_idxs = [
            i for i, (_, label) in enumerate(dataset.samples)
            if label == cls_idx
        ]
        if len(candidates_idxs) >= N:
            sample_idxs = random.sample(candidates_idxs, N)
        else:
            sample_idxs = candidates_idxs  # Use all available
            
        for col, i in enumerate(sample_idxs):
            image, _ = dataset[i]
            image = tensor_to_numpy_image(image, imagenet_mean, imagenet_std)
            ax = axes[row, col]
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])

        axes[row, 0].set_ylabel(cls, fontsize=20)

    plt.tight_layout()
    
    if save_name:
        save_path = PLOTS_DIR / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_name}")
    
    plt.close()


def main():
    """Main function to generate all plots."""
    print("\n1️⃣  Setting up data loaders...")
    
    # Check if dataset exists
    dataset_root = project_root / "dataset"
    if not dataset_root.exists():
        print(f"\n❌ Dataset not found at: {dataset_root}")
        print("   Please run split_dataset.py first to create the dataset.")
        return
    
    # Data transforms
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ]),
    }
    
    # Load datasets
    try:
        image_datasets = {
            x: datasets.ImageFolder(os.path.join(dataset_root, x), data_transforms[x])
            for x in ['train', 'val']
        }
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        print(f"   Make sure dataset is at: {dataset_root}")
        return
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    print(f"   ✓ Train size: {dataset_sizes['train']}")
    print(f"   ✓ Val size: {dataset_sizes['val']}")
    print(f"   ✓ Classes: {len(class_names)}")
    
    # Generate plots
    print("\n2️⃣  Generating distribution plots...")
    
    train_histo = get_dataset_histogram(image_datasets['train'])
    val_histo = get_dataset_histogram(image_datasets['val'])
    
    # Plot 1: Distribution (frequency)
    plot_dataset_histo(
        train_histo, val_histo,
        as_dist=True,
        save_name="01_train_val_distribution.png"
    )
    
    # Plot 2: Histogram (counts)
    plot_dataset_histo(
        train_histo, val_histo,
        as_dist=False,
        y_lim=(0, 2000),
        save_name="02_train_val_histogram.png"
    )
    
    # Plot 3: Training data histogram
    plot_histo(
        train_histo,
        "Training Data Distribution",
        as_dist=True,
        save_name="03_train_distribution.png"
    )
    
    # Plot 4: Validation data histogram
    plot_histo(
        val_histo,
        "Validation Data Distribution",
        as_dist=True,
        save_name="04_val_distribution.png"
    )
    
    print("\n3️⃣  Generating sample visualizations...")
    
    # Plot 5: Random samples from each class
    plot_dataset_sample(
        5, image_datasets['train'], 'train',
        imagenet_mean, imagenet_std,
        save_name="05_random_samples_per_class.png"
    )
    
    print("\n✅ All plots extracted successfully!")
    print(f"\n📁 Plots saved to: {PLOTS_DIR}")
    print("\nGenerated plots:")
    for plot_file in sorted(PLOTS_DIR.glob("*.png")):
        print(f"   - {plot_file.name}")


if __name__ == "__main__":
    main()

