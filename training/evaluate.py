"""
Evaluation and testing script for chess piece classifier.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from model import load_model, get_model
from utils import create_weighted_sampler, load_datasets, get_dataloaders, IMAGENET_MEAN, IMAGENET_STD

def eval_extended(model, dataloader_clean, dataloader_occluded, device, confidence_threshold=0.8):
  model.eval()

  with torch.no_grad():
    corrects = 0.0
    num_samples = 0

    for inputs, labels in tqdm(dataloader_occluded):
      inputs = inputs.to(device)
      labels = labels.to(device)
      outputs = model(inputs)
      probs = torch.softmax(outputs, dim=1)
      confs, preds = probs.max(dim=1)

      correct = preds == labels
      low_conf = confs < confidence_threshold

      accepted = correct | low_conf


      corrects += accepted.sum().item()
      num_samples += labels.size(0)

    for inputs, labels in tqdm(dataloader_clean):
      inputs = inputs.to(device)
      labels = labels.to(device)

      outputs = model(inputs)
      probs = torch.softmax(outputs, dim=1)
      _, preds = probs.max(dim=1)

      corrects += (preds == labels).sum().item()
      num_samples += labels.size(0)

  accuracy = corrects / num_samples
  return accuracy





def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, normalize=False):
    """
    Plot confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure
        normalize: Whether to normalize the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_per_class_metrics(y_true, y_pred, class_names, save_path=None):
    """
    Plot per-class precision, recall, and F1-score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    ax.bar(x - width, precision, width, label='Precision', color='#2E86AB')
    ax.bar(x, recall, width, label='Recall', color='#A23B72')
    ax.bar(x + width, f1, width, label='F1-Score', color='#F18F01')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved per-class metrics to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confidence_distribution(confidences, labels, class_names, save_path=None):
    """
    Plot confidence distribution per class.
    
    Args:
        confidences: Array of confidence scores
        labels: Array of ground truth labels
        class_names: List of class names
        save_path: Path to save figure
    """
    plt.figure(figsize=(14, 6))
    
    for i, class_name in enumerate(class_names):
        class_confs = confidences[labels == i]
        if len(class_confs) > 0:
            plt.hist(class_confs, bins=50, alpha=0.5, label=class_name)
    
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Confidence Distribution by Class', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confidence distribution to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_ood_detection(model, clean_dataloader, occluded_dataloader, 
                          device, save_dir=None):
    """
    Analyze OOD detection using confidence thresholding.
    
    Args:
        model: PyTorch model
        clean_dataloader: DataLoader for clean images
        occluded_dataloader: DataLoader for occluded images
        device: Device to run on
        save_dir: Directory to save plots
        
    Returns:
        clean_confs: Confidence scores for clean images
        occluded_confs: Confidence scores for occluded images
    """
    model.eval()
    
    clean_confs = []
    occluded_confs = []
    
    print("Analyzing clean images...")
    with torch.no_grad():
        for imgs, _ in tqdm(clean_dataloader):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            confs, _ = probs.max(dim=1)
            clean_confs.extend(confs.cpu().numpy())
    
    print("Analyzing occluded images...")
    with torch.no_grad():
        for imgs, labels in tqdm(occluded_dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            
            # Only consider incorrect predictions on occluded images
            mask = (preds != labels)
            confs_filtered = confs[mask]
            occluded_confs.extend(confs_filtered.cpu().numpy())
    
    clean_confs = np.array(clean_confs)
    occluded_confs = np.array(occluded_confs)
    
    # Plot ECDF
    plt.figure(figsize=(10, 6))
    
    # Calculate and plot ECDF
    clean_sorted = np.sort(clean_confs)
    clean_ecdf = np.arange(1, len(clean_sorted) + 1) / len(clean_sorted)
    
    occluded_sorted = np.sort(occluded_confs)
    occluded_ecdf = np.arange(1, len(occluded_sorted) + 1) / len(occluded_sorted)
    
    plt.plot(clean_sorted, clean_ecdf, linewidth=2, 
             label=f'Clean (n={len(clean_confs)})')
    plt.plot(occluded_sorted, occluded_ecdf, linewidth=2, 
             label=f'Occluded (n={len(occluded_confs)})')
    
    plt.xlim(0, 1)
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title('Confidence Distribution: Clean vs Occluded', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'ood_confidence_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved OOD analysis to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print statistics
    print("\nOOD Detection Analysis:")
    print(f"  Clean confidence: {clean_confs.mean():.4f} ± {clean_confs.std():.4f}")
    print(f"  Occluded confidence: {occluded_confs.mean():.4f} ± {occluded_confs.std():.4f}")
    print(f"  Separation: {clean_confs.mean() - occluded_confs.mean():.4f}")
    
    # Suggest threshold
    suggested_threshold = np.percentile(clean_confs, 5)  # 5th percentile of clean
    print(f"  Suggested threshold (5th percentile of clean): {suggested_threshold:.4f}")
    
    return clean_confs, occluded_confs


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate chess piece classifier')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet50', 'vgg16'],
                       help='Model architecture')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Root directory containing val/ or test/ folder')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of data loading workers')
    parser.add_argument('--balanced', action='store_true',
                        help='test on balanced dataset')
    
    # OOD detection (optional)
    parser.add_argument('--occluded-dir', type=str, default=None,
                       help='Optional directory with occluded images for OOD analysis')
    parser.add_argument('--clean-dir', type=str, default=None,
                       help='Optional directory with clean images for OOD analysis')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, model_name=args.model, device=device)
    print("✓ Model loaded\n")
    
    # Load dataset
    print(f"Loading {args.split} dataset from: {args.data_dir}")
    
    # Create a temporary structure to load single split
    import tempfile
    import shutil
    from pathlib import Path
    
    data_root = Path(args.data_dir)
    split_clean_dir = data_root / (args.split + "_clean")
    split_occluded_dir = data_root / (args.split + "_occluded")

    
    if not split_clean_dir.exists():
        raise ValueError(f"Split directory not found: {split_clean_dir}")
    
    if not split_occluded_dir.exists():
        raise ValueError(f"Split directory not found: {split_occluded_dir}")
    
    # Use the existing split directory structure
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    dataset_clean = datasets.ImageFolder(str(split_clean_dir), transform=transform)
    dataset_occluded = datasets.ImageFolder(str(split_occluded_dir), transform=transform)

    if args.balanced: 
        dataloader_clean = torch.utils.data.DataLoader(
            dataset_clean, batch_size=args.batch_size, 
            sampler=create_weighted_sampler(dataset_clean), num_workers=args.num_workers
        )
        dataloader_occluded = torch.utils.data.DataLoader(
            dataset_occluded, batch_size=args.batch_size, 
            sampler=create_weighted_sampler(dataset_occluded), num_workers=args.num_workers
        )
    else: 
        dataloader_clean = torch.utils.data.DataLoader(
            dataset_clean, batch_size=args.batch_size, 
            shuffle=False, num_workers=args.num_workers
        )
        dataloader_occluded = torch.utils.data.DataLoader(
            dataset_occluded, batch_size=args.batch_size, 
            shuffle=False, num_workers=args.num_workers
        )

    
    class_names = dataset_clean.classes
    print(f"  Classes: {len(class_names)}")
    print(f"  Samples: {len(dataset_clean)}\n")
    
    # Evaluate
    print("Evaluating model...")
    accuracy = eval_extended(
        model, dataloader_clean,dataloader_occluded, device
    )
    
    print(f"\n✅ Overall Accuracy: {accuracy:.4f}")
    
  
    # OOD analysis (if provided)
    if args.clean_dir and args.occluded_dir:
        print("\nPerforming OOD detection analysis...")
        
        clean_dataset = datasets.ImageFolder(args.clean_dir, transform=transform)
        occluded_dataset = datasets.ImageFolder(args.occluded_dir, transform=transform)
        
        clean_loader = torch.utils.data.DataLoader(
            clean_dataset, batch_size=args.batch_size, 
            shuffle=False, num_workers=args.num_workers
        )
        occluded_loader = torch.utils.data.DataLoader(
            occluded_dataset, batch_size=args.batch_size, 
            shuffle=False, num_workers=args.num_workers
        )
        
        analyze_ood_detection(model, clean_loader, occluded_loader, device, args.output_dir)
    
    print(f"\n✅ Evaluation complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

