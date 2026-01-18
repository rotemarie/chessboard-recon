"""
Utility functions for data loading, visualization, and helpers.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_data_transforms(augment_train=True):
    """
    Get data transformations for training and validation.
    
    Args:
        augment_train: Whether to apply data augmentation to training data
        
    Returns:
        dict: Dictionary with 'train' and 'val' transforms
    """
    if augment_train:
        train_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    return {'train': train_transforms, 'val': val_transforms}


def load_datasets(data_dir, augment_train=True):
    """
    Load training and validation datasets.
    
    Args:
        data_dir: Root directory containing 'train' and 'val' folders
        augment_train: Whether to apply data augmentation
        
    Returns:
        image_datasets: Dictionary of datasets
        dataset_sizes: Dictionary of dataset sizes
        class_names: List of class names
    """
    data_transforms = get_data_transforms(augment_train)
    
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    return image_datasets, dataset_sizes, class_names


def create_weighted_sampler(dataset, num_samples=None):
    """
    Create a weighted random sampler to balance class distribution.
    
    Args:
        dataset: PyTorch dataset
        num_samples: Number of samples per epoch (None = full dataset size)
        
    Returns:
        WeightedRandomSampler: Sampler for DataLoader
    """
    # Get class counts
    counts = Counter(dataset.targets)
    class_counts = np.array([counts[i] for i in range(len(counts))])
    
    # Calculate inverse frequency weights
    weights = 1.0 / class_counts
    
    # Create a list of weights for every single image
    sample_weights = weights[dataset.targets]
    sample_weights = torch.from_numpy(sample_weights).double()
    
    # Create the sampler
    if num_samples is None:
        num_samples = len(dataset)
        
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=num_samples,
        replacement=True  # Sample with replacement to ensure equal probability
    )
    
    return sampler


def get_dataloaders(image_datasets, batch_size=16, num_workers=2, 
                    use_weighted_sampler=True, num_samples=None):
    """
    Create DataLoaders for training and validation.
    
    Args:
        image_datasets: Dictionary of datasets
        batch_size: Batch size
        num_workers: Number of worker processes
        use_weighted_sampler: Whether to use weighted sampling for balancing
        num_samples: Number of samples per epoch (None = auto-calculate)
        
    Returns:
        dataloaders: Dictionary of DataLoaders
    """
    if use_weighted_sampler:
        # Auto-calculate num_samples if not provided
        if num_samples is None:
            # Calculate balanced number of samples
            train_histo = get_dataset_histogram(image_datasets['train'])
            num_classes = len(image_datasets['train'].classes)
            # Average samples per class (excluding empty class for balance)
            avg_without_empty = (sum(train_histo.values()) - train_histo.get('empty', 0)) // (num_classes - 1)
            num_samples = avg_without_empty * num_classes
        
        dataloaders = {
            x: DataLoader(
                image_datasets[x],
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=create_weighted_sampler(
                    image_datasets[x],
                    num_samples if x == 'train' else None
                )
            )
            for x in ['train', 'val']
        }
    else:
        dataloaders = {
            x: DataLoader(
                image_datasets[x],
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True
            )
            for x in ['train', 'val']
        }
    
    return dataloaders


def get_dataset_histogram(dataset):
    """
    Get class distribution histogram from dataset.
    
    Args:
        dataset: PyTorch dataset
        
    Returns:
        dict: Histogram of class counts
    """
    idx_to_class = {idx: cls for cls, idx in dataset.class_to_idx.items()}
    counts = Counter(dataset.targets)
    histogram = {idx_to_class[idx]: count for idx, count in counts.items()}
    return histogram


def tensor_to_numpy_image(image, mean=None, std=None):
    """
    Transform tensor to displayable numpy image.
    
    Args:
        image: Tensor image (C, H, W)
        mean: Mean for denormalization
        std: Standard deviation for denormalization
        
    Returns:
        numpy array: Image in (H, W, C) format
    """
    if mean is None:
        mean = np.array(IMAGENET_MEAN)
    if std is None:
        std = np.array(IMAGENET_STD)
    
    image = image.numpy().transpose((1, 2, 0))
    image = std * image + mean
    image = image.clip(0, 1)
    return image


def plot_dataset_sample(N, dataset, dataset_name='Dataset', save_path=None):
    """
    Plot N random samples from each class.
    
    Args:
        N: Number of samples per class
        dataset: PyTorch dataset
        dataset_name: Name for the plot title
        save_path: Optional path to save the figure
    """
    num_classes = len(dataset.classes)
    fig, axes = plt.subplots(
        num_classes, N,
        figsize=(N * 2, num_classes * 2)
    )
    fig.suptitle(f'Random Samples from Each Class - {dataset_name}', 
                 y=1.0, fontsize=25)
    
    for row, cls in enumerate(dataset.classes):
        cls_idx = dataset.class_to_idx[cls]
        candidates_idxs = [
            i for i, (_, label) in enumerate(dataset.samples)
            if label == cls_idx
        ]
        
        if len(candidates_idxs) >= N:
            sample_idxs = random.sample(candidates_idxs, N)
        else:
            sample_idxs = candidates_idxs
            
        for col, i in enumerate(sample_idxs):
            image, _ = dataset[i]
            image = tensor_to_numpy_image(image)
            ax = axes[row, col]
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
        
        axes[row, 0].set_ylabel(cls, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_model_predictions(model, dataloader, idx_to_class, device, 
                                 num_images=10, confidence_threshold=None):
    """
    Visualize model predictions on a batch of images.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        idx_to_class: Dictionary mapping class indices to names
        device: Device to run inference on
        num_images: Number of images to visualize
        confidence_threshold: Optional confidence threshold for OOD detection
    """
    model.eval()
    
    images_to_plot = []
    pred_labels_to_plot = []
    gt_labels_to_plot = []
    confidences = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)
            
            # Apply confidence threshold if provided
            if confidence_threshold is not None:
                preds = preds.clone()
                preds[confs < confidence_threshold] = -1
                
                # Add "occluded" class
                idx_to_class_with_ood = idx_to_class.copy()
                idx_to_class_with_ood[-1] = "occluded"
            else:
                idx_to_class_with_ood = idx_to_class
            
            pred_labels_cpu = [idx_to_class_with_ood[p.item()] for p in preds]
            gt_labels_cpu = [idx_to_class[l.item()] for l in labels]
            
            images_to_plot.extend(inputs.cpu())
            pred_labels_to_plot.extend(pred_labels_cpu)
            gt_labels_to_plot.extend(gt_labels_cpu)
            confidences.extend(confs.cpu().numpy())
            
            if len(images_to_plot) >= num_images:
                break
    
    # Plot
    cols = min(num_images, 5)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (img_t, pred_label, gt_label, conf) in enumerate(
        zip(images_to_plot[:num_images], 
            pred_labels_to_plot[:num_images], 
            gt_labels_to_plot[:num_images],
            confidences[:num_images])
    ):
        img = tensor_to_numpy_image(img_t)
        
        axes[i].imshow(img)
        title = f"P: {pred_label}\nGT: {gt_label}"
        if confidence_threshold is not None:
            title += f"\nConf: {conf:.2f}"
        
        color = 'green' if pred_label == gt_label else 'red'
        axes[i].set_title(title, fontsize=9, color=color)
        axes[i].axis("off")
    
    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.show()


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test data transforms
    transforms_dict = get_data_transforms()
    print("✓ Data transforms created")
    
    # Test tensor to image conversion
    dummy_tensor = torch.randn(3, 224, 224)
    img = tensor_to_numpy_image(dummy_tensor)
    print(f"✓ Tensor to image conversion: {img.shape}")
    
    print("\n✓ All utilities tests passed!")

