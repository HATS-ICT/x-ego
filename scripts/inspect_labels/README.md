# Enemy Location Nowcast - Label Inspection Tools

This directory contains tools for analyzing the label distribution and computing class weights for the enemy location nowcast task.

## Overview

The enemy location nowcast task predicts the locations of 5 enemy players in a multi-label classification setting. The dataset exhibits **severe class imbalance** with a ratio of **124.7:1** (most common vs least common location).

## Scripts

### 1. `inspect_enemy_location_nowcast.py`

**Purpose**: Analyzes the label distribution to identify class imbalance issues.

**Usage**:
```bash
python scripts/inspect_labels/inspect_enemy_location_nowcast.py
```

**Outputs**:
- **Console Report**: Detailed statistics about label distribution
- **Visualizations** (saved to `artifacts/label_analysis/`):
  - `location_distribution_*.png` - Bar plots (linear & log scale)
  - `cumulative_distribution_*.png` - Cumulative distribution plots
- **Summary Document**: `artifacts/label_analysis/ANALYSIS_SUMMARY.md`

**Key Findings**:
- 23 unique location classes
- Severe class imbalance (ratio 124.7:1)
- Top 3 classes account for 33.8% of data
- Bottom 3 classes account for only 1.43% of data
- Most common: CTSpawn (14.3%)
- Least common: Scaffolding (0.11%)

---

### 2. `compute_class_weights.py`

**Purpose**: Computes class weights to balance the loss function during training.

**Usage**:
```bash
python scripts/inspect_labels/compute_class_weights.py
```

**Outputs**:
- **Weight files** (saved to `artifacts/class_weights/`):
  - `class_weights_{partition}_{method}.json` for each partition and method

**Methods Available**:
1. **`inverse`**: Most aggressive weighting (weight = 1/count)
   - Best for: Extreme imbalance, when you want rare classes to dominate
   - Weight ratio: 114.7x
   
2. **`inverse_sqrt`**: Moderate weighting (weight = 1/√count) ⭐ **RECOMMENDED**
   - Best for: Balanced approach between common and rare classes
   - Weight ratio: 10.7x
   - Less aggressive than inverse, more stable training
   
3. **`effective_num`**: Research-based method (Cui et al., 2019)
   - Best for: Extreme imbalance with many overlapping samples
   - Weight ratio: 56.3x
   - Based on "Class-Balanced Loss Based on Effective Number of Samples"
   
4. **`pos_weight`**: For BCEWithLogitsLoss
   - Best for: Direct use with PyTorch's BCEWithLogitsLoss
   - Computes pos_weight = neg_samples / pos_samples

---

## Integration Guide

### Step 1: Load Class Weights

Add this to your model initialization (e.g., in `models/multi_agent_location_predictor.py`):

```python
import json
from pathlib import Path

class MultiAgentEnemyLocationPredictionModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        # ... existing initialization ...
        
        # Load class weights for multi-label classification
        if self.task_form == 'multi-label-cls':
            self.class_weights = self._load_class_weights(cfg)
    
    def _load_class_weights(self, cfg):
        """Load pre-computed class weights."""
        # Path to weights file (change method as needed)
        weights_path = Path('artifacts/class_weights/class_weights_train_inverse_sqrt.json')
        
        if not weights_path.exists():
            self.print(f"Warning: Class weights not found at {weights_path}")
            return None
        
        with open(weights_path, 'r') as f:
            data = json.load(f)
            weights_dict = data['weights']
        
        # Convert to tensor matching label order
        # Assuming self.place_names is your ordered list of locations
        weights = torch.tensor(
            [weights_dict[place] for place in self.place_names],
            dtype=torch.float32
        )
        
        self.print(f"Loaded class weights from {weights_path}")
        self.print(f"Weight range: {weights.min():.4f} to {weights.max():.4f}")
        
        return weights
```

### Step 2: Use Weights in Loss Function

Modify your loss computation:

```python
def compute_loss(self, predictions, targets):
    """
    Compute weighted loss for multi-label classification.
    
    Args:
        predictions: [batch_size, num_agents, num_classes] logits
        targets: [batch_size, num_agents, num_classes] binary labels
    """
    if self.task_form == 'multi-label-cls':
        # Flatten predictions and targets
        pred_flat = predictions.view(-1, predictions.shape[-1])  # [B*A, C]
        target_flat = targets.view(-1, targets.shape[-1])        # [B*A, C]
        
        # Option 1: Use BCEWithLogitsLoss with pos_weight
        if self.class_weights is not None:
            loss = F.binary_cross_entropy_with_logits(
                pred_flat, 
                target_flat,
                pos_weight=self.class_weights.to(pred_flat.device)
            )
        else:
            loss = F.binary_cross_entropy_with_logits(pred_flat, target_flat)
        
        # Option 2: Manual weighted loss
        # bce_loss = F.binary_cross_entropy_with_logits(
        #     pred_flat, target_flat, reduction='none'
        # )
        # weights_expanded = self.class_weights.unsqueeze(0).to(pred_flat.device)
        # weighted_loss = (bce_loss * weights_expanded).mean()
        
        return loss
```

### Step 3: Enhanced Evaluation Metrics

Add per-class metrics to your validation/test step:

```python
from torchmetrics.classification import (
    MultilabelHammingDistance,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall
)

def __init__(self, cfg):
    # ... existing init ...
    
    if self.task_form == 'multi-label-cls':
        num_labels = cfg.data.num_places
        
        # Overall metrics
        self.hamming_dist = MultilabelHammingDistance(num_labels=num_labels)
        
        # Per-class metrics
        self.f1_per_class = MultilabelF1Score(
            num_labels=num_labels, 
            average='none'  # Get per-class scores
        )
        
        # Macro-averaged (equal weight to all classes)
        self.f1_macro = MultilabelF1Score(
            num_labels=num_labels,
            average='macro'
        )
        
        # Micro-averaged (overall performance)
        self.f1_micro = MultilabelF1Score(
            num_labels=num_labels,
            average='micro'
        )

def validation_step(self, batch, batch_idx):
    # ... compute predictions ...
    
    # Overall metrics
    hamming = self.hamming_dist(preds, targets)
    f1_macro = self.f1_macro(preds, targets)
    f1_micro = self.f1_micro(preds, targets)
    
    # Per-class metrics
    f1_per_class = self.f1_per_class(preds, targets)
    
    # Log overall metrics
    self.log('val/hamming_distance', hamming)
    self.log('val/f1_macro', f1_macro)
    self.log('val/f1_micro', f1_micro)
    
    # Log per-class F1 scores for monitoring
    for i, place_name in enumerate(self.place_names):
        self.log(f'val/f1_{place_name}', f1_per_class[i])
    
    # Identify worst performing classes
    worst_classes = torch.topk(f1_per_class, k=5, largest=False)
    print(f"\nWorst 5 classes:")
    for idx in worst_classes.indices:
        print(f"  {self.place_names[idx]}: F1={f1_per_class[idx]:.4f}")
```

### Step 4: Monitor Training

Key metrics to track:
- **Hamming Distance**: Lower is better, but biased toward common classes
- **Macro F1**: Equal weight to all classes - **use this as primary metric**
- **Micro F1**: Overall performance across all predictions
- **Per-class F1**: Monitor rare classes (Scaffolding, Ladder, Stairs)

### Step 5: Hyperparameter Tuning

Experiment with different weighting methods:

```yaml
# In your config file
data:
  class_weight_method: 'inverse_sqrt'  # Options: inverse, inverse_sqrt, effective_num, pos_weight
  class_weight_scale: 1.0              # Scale factor to adjust weight strength
```

## Recommendations

### 🔴 Critical Actions
1. ✅ **Use class weights** - Start with `inverse_sqrt` method
2. ✅ **Track macro F1** - Don't rely only on Hamming distance
3. ✅ **Monitor rare classes** - Check if model ever predicts Scaffolding, Ladder, etc.

### 🟡 Important Considerations
1. **Focal Loss**: Consider implementing for extreme imbalance
   ```python
   def focal_loss(pred, target, alpha=0.25, gamma=2.0):
       bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
       pt = torch.exp(-bce)
       focal = alpha * (1 - pt) ** gamma * bce
       return focal.mean()
   ```

2. **Data Augmentation**: Oversample samples with rare locations

3. **Hierarchical Classification**: Group locations by map regions
   - T Side, CT Side, Bombsites, Mid/Connectors, Side Areas
   - Predict region first, then specific location

4. **Ensemble Methods**: Combine predictions from multiple models
   - One model trained with `inverse_sqrt` weights
   - Another with `effective_num` weights
   - Ensemble their predictions

### 🟢 Optional Enhancements
1. **Stratified Sampling**: Ensure each batch has diverse locations
2. **Curriculum Learning**: Start with common classes, gradually include rare ones
3. **Post-processing**: Apply game knowledge (e.g., teammates shouldn't be at same location)

## Expected Results

After implementing class weighting:

**Before** (no weighting):
- Hamming Distance: ~0.15
- Macro F1: ~0.30 (poor)
- Rare classes: F1 ≈ 0 (never predicted)

**After** (with inverse_sqrt weighting):
- Hamming Distance: ~0.18 (slightly worse, but expected)
- Macro F1: ~0.50-0.60 (much better)
- Rare classes: F1 > 0.1 (some predictions)

The goal is to improve **macro F1** and ensure rare classes get predicted, even if overall Hamming distance increases slightly.

## Files Generated

```
artifacts/
├── label_analysis/
│   ├── ANALYSIS_SUMMARY.md
│   ├── location_distribution_train.png
│   ├── location_distribution_val.png
│   ├── location_distribution_test.png
│   ├── location_distribution_overall.png
│   ├── cumulative_distribution_train.png
│   ├── cumulative_distribution_val.png
│   ├── cumulative_distribution_test.png
│   └── cumulative_distribution_overall.png
│
└── class_weights/
    ├── class_weights_train_inverse.json
    ├── class_weights_train_inverse_sqrt.json      ⭐ RECOMMENDED
    ├── class_weights_train_effective_num.json
    ├── class_weights_train_pos_weight.json
    ├── class_weights_val_*.json
    ├── class_weights_test_*.json
    └── class_weights_all_*.json
```

## References

1. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
2. **Class-Balanced Loss**: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
3. **Multi-Label Classification**: Zhang & Zhou, "A Review on Multi-Label Learning Algorithms", TKDE 2014

## Next Steps

1. Run the inspection scripts to generate analysis and weights
2. Integrate class weights into your training pipeline
3. Add per-class evaluation metrics
4. Retrain your model with weighted loss
5. Compare results with baseline (no weighting)
6. Experiment with different weighting methods
7. Consider advanced techniques (focal loss, hierarchical classification)

---

**Created by**: Enemy Location Nowcast Analysis Tools  
**Last Updated**: After dataset analysis of 16,789 samples

