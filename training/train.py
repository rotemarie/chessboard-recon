"""
Training script for chess piece classification.
"""

import argparse
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

try:
    import comet_ml
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("⚠️  comet_ml not available. Training will proceed without experiment tracking.")

from model import get_model
from utils import load_datasets, get_dataloaders, set_seed


def train_model(model, criterion, optimizer, scheduler, dataloaders, device,
                experiment=None, num_epochs=25, patience=10, log_per_batch_metrics=False,
                checkpoint_dir=None):
    """
    Train the model with early stopping.
    
    Args:
        model: PyTorch model
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        dataloaders: Dictionary of DataLoaders
        device: Device to train on
        experiment: Comet.ml experiment (optional)
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        log_per_batch_metrics: Whether to log metrics per batch
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        model: Trained model
        history: Training history
    """
    since = time.time()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    with TemporaryDirectory() as tempdir:
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            best_model_params_path = os.path.join(checkpoint_dir, 'best_model.pth')
        else:
            best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
            
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        epochs_without_improvement = 0  # Counter for early stopping
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 40)
            
            for phase in ['train', 'val']:
                epoch_size = 0
                
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                
                running_loss = 0.0
                running_corrects = 0
                
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    # Log per-batch metrics if requested
                    if log_per_batch_metrics and experiment:
                        batch_acc = (preds == labels.data).double().mean().item()
                        experiment.log_metrics({
                            f"{phase}_batch_loss": loss.item(),
                            f"{phase}_batch_acc": batch_acc
                        })
                    
                    epoch_size += inputs.size(0)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train':
                    scheduler.step()
                
                # Calculate metrics for the epoch
                epoch_loss = running_loss / epoch_size
                epoch_acc = running_corrects.double() / epoch_size
                
                # Log to Comet.ml
                if experiment:
                    experiment.log_metrics({
                        f"{phase}_loss": epoch_loss,
                        f"{phase}_acc": epoch_acc
                    }, step=epoch)
                
                # Save to history
                history[f'{phase}_loss'].append(epoch_loss)
                history[f'{phase}_acc'].append(epoch_acc.item())
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # Save best model
                if phase == 'val':
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        epochs_without_improvement = 0
                        torch.save(model.state_dict(), best_model_params_path)
                        print(f'  → New best model saved! (Acc: {best_acc:.4f})')
                    else:
                        epochs_without_improvement += 1
            
            print()
            
            # Check if patience limit is reached
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement!")
                break
        
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')
        
        # Load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
        
        # Save final checkpoint
        if checkpoint_dir:
            final_path = os.path.join(checkpoint_dir, 'final_model.pth')
            torch.save(model.state_dict(), final_path)
            print(f'Final model saved to: {final_path}')
    
    return model, history


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train chess piece classifier')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Root directory containing train/ and val/ folders')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of data loading workers (default: 2)')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--no-weighted-sampler', action='store_true',
                       help='Disable weighted sampling for class balancing')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet50', 'vgg16'],
                       help='Model architecture (default: resnet18)')
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='Freeze backbone for transfer learning')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Do not use pretrained weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum number of epochs (default: 100)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum (default: 0.9)')
    parser.add_argument('--step-size', type=int, default=7,
                       help='LR scheduler step size (default: 7)')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='LR scheduler gamma (default: 0.1)')
    
    # Experiment tracking
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Comet.ml experiment name')
    parser.add_argument('--log-per-batch', action='store_true',
                       help='Log metrics per batch (slower)')
    
    # Output arguments
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    
    # Load datasets
    print("\nLoading datasets...")
    image_datasets, dataset_sizes, class_names = load_datasets(
        args.data_dir,
        augment_train=not args.no_augmentation
    )
    
    print(f"  Train size: {dataset_sizes['train']}")
    print(f"  Val size: {dataset_sizes['val']}")
    print(f"  Classes: {len(class_names)}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = get_dataloaders(
        image_datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=not args.no_weighted_sampler
    )
    
    # Create model
    print(f"\nCreating model: {args.model}")
    model = get_model(
        model_name=args.model,
        num_classes=len(class_names),
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone
    )
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    
    if args.freeze_backbone:
        # Only optimize final layer parameters
        if args.model == 'vgg16':
            optimizer = optim.SGD(model.classifier.parameters(), 
                                lr=args.lr, momentum=args.momentum)
        else:
            optimizer = optim.SGD(model.fc.parameters(), 
                                lr=args.lr, momentum=args.momentum)
    else:
        # Optimize all parameters
        optimizer = optim.SGD(model.parameters(), 
                            lr=args.lr, momentum=args.momentum)
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    # Setup Comet.ml experiment
    experiment = None
    if COMET_AVAILABLE:
        try:
            experiment = comet_ml.start()
            if args.experiment_name:
                experiment.set_name(args.experiment_name)
            experiment.auto_metric_logging = False
            experiment.log_parameters(vars(args))
            print("\n✓ Comet.ml experiment initialized")
        except Exception as e:
            print(f"\n⚠️  Could not initialize Comet.ml: {e}")
            experiment = None
    
    # Train model
    print("\nStarting training...")
    print("="*60)
    
    model, history = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloaders=dataloaders,
        device=device,
        experiment=experiment,
        num_epochs=args.epochs,
        patience=args.patience,
        log_per_batch_metrics=args.log_per_batch,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # End experiment
    if experiment:
        experiment.end()
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()

