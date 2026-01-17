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
from utils import load_datasets, get_dataloaders, IMAGENET_MEAN, IMAGENET_STD


def evaluate_model(model, dataloader, device, return_predictions=False):
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run on
        return_predictions: Whether to return predictions and labels
        
    Returns:
        accuracy: Overall accuracy
        predictions: (optional) List of predictions
        labels: (optional) List of ground truth labels
        confidences: (optional) List of confidence scores
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_confs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    if return_predictions:
        return accuracy, np.array(all_preds), np.array(all_labels), np.array(all_confs)
    else:
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
    split_dir = data_root / args.split
    
    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")
    
    # Use the existing split directory structure
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    dataset = datasets.ImageFolder(str(split_dir), transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=args.num_workers
    )
    
    class_names = dataset.classes
    print(f"  Classes: {len(class_names)}")
    print(f"  Samples: {len(dataset)}\n")
    
    # Evaluate
    print("Evaluating model...")
    accuracy, predictions, labels, confidences = evaluate_model(
        model, dataloader, device, return_predictions=True
    )
    
    print(f"\n✅ Overall Accuracy: {accuracy:.4f}")
    print(f"   Mean Confidence: {confidences.mean():.4f}\n")
    
    # Print classification report
    print("Classification Report:")
    print("=" * 80)
    report = classification_report(labels, predictions, target_names=class_names, digits=4)
    print(report)
    
    # Save classification report
    report_path = Path(args.output_dir) / f'{args.split}_classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"Mean Confidence: {confidences.mean():.4f}\n\n")
        f.write(report)
    print(f"✓ Saved report to: {report_path}\n")
    
    # Plot confusion matrix
    print("Generating confusion matrix...")
    plot_confusion_matrix(
        labels, predictions, class_names,
        save_path=Path(args.output_dir) / f'{args.split}_confusion_matrix.png'
    )
    
    # Plot normalized confusion matrix
    plot_confusion_matrix(
        labels, predictions, class_names,
        save_path=Path(args.output_dir) / f'{args.split}_confusion_matrix_normalized.png',
        normalize=True
    )
    
    # Plot per-class metrics
    print("Generating per-class metrics plot...")
    plot_per_class_metrics(
        labels, predictions, class_names,
        save_path=Path(args.output_dir) / f'{args.split}_per_class_metrics.png'
    )
    
    # Plot confidence distribution
    print("Generating confidence distribution...")
    plot_confidence_distribution(
        confidences, labels, class_names,
        save_path=Path(args.output_dir) / f'{args.split}_confidence_distribution.png'
    )
    
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

