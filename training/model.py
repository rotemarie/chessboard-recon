"""
Model architectures for chess piece classification.

This project uses ResNet18 fine-tuning as the primary method (89.08% val accuracy).
Other architectures (ResNet50, VGG16) are available but ResNet18 performed best.
"""

import torch
import torch.nn as nn
from torchvision import models


def get_model(model_name='resnet18', num_classes=13, pretrained=True, freeze_backbone=False):
    """
    Get a model for chess piece classification.
    
    Args:
        model_name: Name of the architecture. Options:
            - 'resnet18' (recommended - used in this project, 89% accuracy)
            - 'resnet50' (experimental - larger model, 23M params)
            - 'vgg16' (experimental - much larger, 138M params)
        num_classes: Number of classes (default: 13 for chess pieces)
        pretrained: Whether to use pretrained ImageNet weights (recommended: True)
        freeze_backbone: Whether to freeze the backbone for transfer learning
                        (recommended: False for fine-tuning)
        
    Returns:
        model: PyTorch model ready for training/inference
        
    Example:
        # ResNet18 fine-tuning (method used in project)
        model = get_model('resnet18', num_classes=13, pretrained=True)
    """
    if model_name == 'resnet18':
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None
        model = models.resnet18(weights=weights)
        
        # Freeze backbone if transfer learning
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        
        # Replace final layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'resnet50':
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
        else:
            weights = None
        model = models.resnet50(weights=weights)
        
        # Freeze backbone if transfer learning
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        
        # Replace final layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'vgg16':
        if pretrained:
            weights = models.VGG16_Weights.DEFAULT
        else:
            weights = None
        model = models.vgg16(weights=weights)
        
        # Freeze convolutional backbone if transfer learning
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        
        # Replace final classifier layer
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def load_model(model_path, model_name='resnet18', num_classes=13, device='cpu'):
    """
    Load a trained model from a checkpoint.
    
    Args:
        model_path: Path to the model checkpoint (.pth file)
        model_name: Name of the architecture
        num_classes: Number of classes
        device: Device to load the model on
        
    Returns:
        model: Loaded PyTorch model
    """
    # Create model architecture
    model = get_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    return model


def count_parameters(model):
    """
    Count the number of trainable and total parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # ResNet18 - Fine-tuning
    model_ft = get_model('resnet18', num_classes=13, pretrained=True, freeze_backbone=False)
    total, trainable = count_parameters(model_ft)
    print(f"\nResNet18 (Fine-tuning):")
    print(f"  Total params: {total:,}")
    print(f"  Trainable params: {trainable:,}")
    
    # ResNet18 - Transfer Learning
    model_tl = get_model('resnet18', num_classes=13, pretrained=True, freeze_backbone=True)
    total, trainable = count_parameters(model_tl)
    print(f"\nResNet18 (Transfer Learning):")
    print(f"  Total params: {total:,}")
    print(f"  Trainable params: {trainable:,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model_ft(dummy_input)
    print(f"\nForward pass test:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    
    print("\n✓ Model creation successful!")

