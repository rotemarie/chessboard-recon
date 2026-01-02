# Out-of-Distribution (OOD) Detection Guide

## Why OOD Detection is Critical

Your project requires handling **occlusions** (hands blocking pieces, pieces being moved, etc.). When a square is occluded, the system should output `"unknown"` instead of making a wrong prediction.

Standard classifiers are **overconfident** - they assign high probabilities even to images they've never seen before. OOD detection methods identify when the input is fundamentally different from training data.

---

## OOD Detection Methods Overview

### 1. ⭐ Baseline: Maximum Softmax Probability (MSP)

**How it works:**
```python
probs = F.softmax(logits, dim=1)
max_prob, pred_class = torch.max(probs, dim=1)

if max_prob < threshold:  # e.g., 0.7
    return "unknown"
else:
    return pred_class
```

**Pros:**
- ✅ Zero implementation cost
- ✅ No additional training required
- ✅ Fast inference

**Cons:**
- ❌ Neural networks are poorly calibrated
- ❌ High confidence on OOD samples
- ❌ Hard to choose threshold

**When to use:** As a baseline to compare against

**Paper:** N/A (standard approach)

---

### 2. ⭐⭐ ODIN: Out-of-Distribution Detector for Neural Networks

**How it works:**
1. Add temperature scaling to softmax
2. Add small perturbation to input
3. Use modified confidence score

```python
# Temperature scaling
scaled_logits = logits / temperature  # temperature > 1

# Input perturbation
loss = F.cross_entropy(scaled_logits, pred_label)
gradient = torch.autograd.grad(loss, input_image)[0]
perturbed_input = input_image - epsilon * torch.sign(gradient)

# Re-compute with perturbed input
new_logits = model(perturbed_input)
new_probs = F.softmax(new_logits / temperature, dim=1)
max_prob, _ = torch.max(new_probs, dim=1)

if max_prob < threshold:
    return "unknown"
```

**Pros:**
- ✅ Significantly better than MSP
- ✅ No retraining required
- ✅ Works with any trained model
- ✅ Well-tested and popular

**Cons:**
- ❌ Two hyperparameters to tune (temperature, epsilon)
- ❌ Requires gradient computation (slightly slower)
- ❌ May need separate threshold per class

**When to use:** Good default choice, proven effectiveness

**Paper:** [Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks](https://arxiv.org/abs/1706.02690) (ICLR 2018)

**Hyperparameters:**
- Temperature: 1000 (typical starting point)
- Epsilon: 0.0014 (typical for images normalized to [0,1])
- Tune on validation set with synthetic occlusions

---

### 3. ⭐⭐⭐ Mahalanobis Distance-Based Detection

**How it works:**
1. Extract features from a hidden layer (e.g., before final FC layer)
2. Compute class-conditional Gaussian distributions during training
3. At inference, measure distance to nearest class distribution

```python
# During training: compute statistics
for class_idx in range(num_classes):
    class_features = features[labels == class_idx]
    class_means[class_idx] = class_features.mean(dim=0)
    # Compute covariance (tied or per-class)

# At inference: compute Mahalanobis distance
feature = model.get_features(input_image)
distances = []
for class_idx in range(num_classes):
    diff = feature - class_means[class_idx]
    # Mahalanobis distance: sqrt(diff^T * Σ^{-1} * diff)
    dist = torch.sqrt(diff @ inv_covariance @ diff.T)
    distances.append(dist)

min_distance = min(distances)
if min_distance > threshold:
    return "unknown"
```

**Pros:**
- ✅ Theoretically principled
- ✅ Very effective in practice
- ✅ Can detect subtle OOD samples
- ✅ Well-suited for feature-based analysis

**Cons:**
- ❌ Need to store class statistics (means, covariance)
- ❌ Computing/inverting covariance can be expensive
- ❌ Requires running full training set through model once
- ❌ More complex implementation

**When to use:** When you want strong performance and can afford complexity

**Paper:** [A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks](https://arxiv.org/abs/1807.03888) (NeurIPS 2018)

**Implementation tips:**
- Use tied covariance (shared across classes) for stability
- Extract features from penultimate layer
- Can combine with input preprocessing like ODIN

---

### 4. ⭐⭐ Energy-Based OOD Detection

**How it works:**
Use energy score instead of softmax probability:

```python
# Energy score (lower = more in-distribution)
energy = -torch.logsumexp(logits, dim=1)

if energy > threshold:
    return "unknown"
```

**Pros:**
- ✅ Simple to implement
- ✅ Better theoretical properties than softmax
- ✅ No additional training required
- ✅ Effective in practice

**Cons:**
- ❌ Still requires threshold tuning
- ❌ May need temperature scaling

**When to use:** Simple and effective alternative to MSP

**Paper:** [Energy-based Out-of-distribution Detection](https://arxiv.org/abs/2010.03759) (NeurIPS 2020)

---

### 5. OpenMax

**How it works:**
Replace final softmax layer with OpenMax layer that models the probability of unknown classes using Extreme Value Theory (EVT).

**Pros:**
- ✅ Principled open-set recognition
- ✅ Can reject unknowns explicitly

**Cons:**
- ❌ Requires modifying model architecture
- ❌ More complex to implement
- ❌ Requires retraining

**When to use:** For research or when you need maximum performance

**Paper:** [Towards Open Set Deep Networks](https://arxiv.org/abs/1511.06233) (CVPR 2016)

---

## Recommendation for Your Project

### 🥇 Primary Recommendation: **Mahalanobis Distance**

**Why:**
- Excellent detection performance
- Works well for small-to-medium datasets
- Fits your use case (detecting occluded pieces)
- Well-documented and tested

**Implementation plan:**
1. Train your classifier normally
2. Extract features from all training samples
3. Compute class means and tied covariance
4. At inference, compute Mahalanobis distance
5. Tune threshold on validation set with synthetic occlusions

### 🥈 Backup Recommendation: **ODIN**

**Why:**
- Simpler to implement
- No need to store statistics
- Still very effective

**Implementation plan:**
1. Train your classifier normally
2. Add temperature scaling and input perturbation
3. Tune temperature and epsilon on validation set
4. Choose threshold

### 🥉 Quick Baseline: **Energy Score**

**Why:**
- Simplest effective method
- Good baseline for comparison

---

## Creating OOD Validation Data

You **must** create synthetic occlusions to tune your OOD detector:

### Occlusion Strategies:

```python
import cv2
import numpy as np

def add_random_occlusion(image):
    """Add realistic occlusions to square images."""
    h, w = image.shape[:2]
    
    # Strategy 1: Random rectangle
    x1, y1 = np.random.randint(0, w//2), np.random.randint(0, h//2)
    x2, y2 = x1 + np.random.randint(w//4, w), y1 + np.random.randint(h//4, h)
    color = np.random.randint(0, 255, 3)
    cv2.rectangle(image, (x1, y1), (x2, y2), color.tolist(), -1)
    
    # Strategy 2: Gaussian blur (hand moving)
    if np.random.random() < 0.3:
        image = cv2.GaussianBlur(image, (15, 15), 0)
    
    # Strategy 3: Random noise
    if np.random.random() < 0.2:
        noise = np.random.randn(h, w, 3) * 50
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    return image
```

### Validation Strategy:

1. Take 10-20% of validation squares
2. Apply occlusions
3. Label them as "OOD"
4. Tune threshold to achieve:
   - High true positive rate on occluded images (>90%)
   - Low false positive rate on normal images (<5%)

---

## Evaluation Metrics

```python
# Compute OOD detection metrics
from sklearn.metrics import roc_auc_score, roc_curve

# Get scores for in-distribution (ID) and OOD samples
id_scores = compute_scores(val_images)
ood_scores = compute_scores(occluded_images)

# Labels: 0 = ID, 1 = OOD
labels = np.concatenate([np.zeros(len(id_scores)), 
                         np.ones(len(ood_scores))])
scores = np.concatenate([id_scores, ood_scores])

# AUROC (higher is better, 1.0 = perfect)
auroc = roc_auc_score(labels, scores)

# FPR at 95% TPR (lower is better)
fpr, tpr, thresholds = roc_curve(labels, scores)
fpr_at_95_tpr = fpr[np.argmax(tpr >= 0.95)]

print(f"AUROC: {auroc:.3f}")
print(f"FPR@95TPR: {fpr_at_95_tpr:.3f}")
```

**Target Performance:**
- AUROC > 0.90 (excellent detection)
- FPR@95TPR < 0.10 (few false alarms)

---

## Implementation Checklist

- [ ] Train base classifier
- [ ] Choose OOD method (Mahalanobis recommended)
- [ ] Create synthetic occlusions for validation
- [ ] Implement OOD detector
- [ ] Tune hyperparameters/threshold on validation set
- [ ] Evaluate on test set with real occlusions
- [ ] Integrate into full pipeline
- [ ] Mark unknown squares as '?' in FEN output

---

## Additional Resources

### Papers to Read:
1. **Mahalanobis:** https://arxiv.org/abs/1807.03888
2. **ODIN:** https://arxiv.org/abs/1706.02690
3. **Energy:** https://arxiv.org/abs/2010.03759
4. **Survey:** https://arxiv.org/abs/2110.11334 (comprehensive overview)

### Code Repositories:
1. **Mahalanobis Official:** https://github.com/pokaxpoka/deep_Mahalanobis_detector
2. **ODIN Official:** https://github.com/facebookresearch/odin
3. **Energy Official:** https://github.com/wetliu/energy_ood

### Blog Posts:
1. https://towardsdatascience.com/out-of-distribution-detection-in-deep-neural-networks-450da9ed7044
2. https://lilianweng.github.io/posts/2022-09-01-ood/

---

## Quick Start Code Template

```python
# ood_detector.py

import torch
import torch.nn.functional as F
import numpy as np

class MahalanobisOODDetector:
    def __init__(self, model, num_classes, feature_dim):
        self.model = model
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # Will be computed from training data
        self.class_means = None
        self.inv_covariance = None
        self.threshold = None
    
    def fit(self, dataloader):
        """Compute class means and covariance from training data."""
        features_list = []
        labels_list = []
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                features = self.model.get_features(images)
                features_list.append(features)
                labels_list.append(labels)
        
        features = torch.cat(features_list)
        labels = torch.cat(labels_list)
        
        # Compute class means
        self.class_means = torch.zeros(self.num_classes, self.feature_dim)
        for c in range(self.num_classes):
            self.class_means[c] = features[labels == c].mean(dim=0)
        
        # Compute tied covariance
        centered = features - self.class_means[labels]
        cov = (centered.T @ centered) / len(features)
        
        # Add regularization for numerical stability
        cov += torch.eye(self.feature_dim) * 1e-6
        self.inv_covariance = torch.inverse(cov)
    
    def compute_distance(self, image):
        """Compute Mahalanobis distance to nearest class."""
        features = self.model.get_features(image)
        
        distances = []
        for c in range(self.num_classes):
            diff = features - self.class_means[c]
            dist = torch.sqrt(diff @ self.inv_covariance @ diff.T)
            distances.append(dist.item())
        
        return min(distances)
    
    def predict_with_ood(self, image):
        """Predict class or return 'unknown' if OOD."""
        distance = self.compute_distance(image)
        
        if distance > self.threshold:
            return "unknown", distance
        else:
            logits = self.model(image)
            pred = torch.argmax(logits, dim=1)
            return pred.item(), distance
```

Good luck! 🎯

