# Decision Tree & Random Forest from Scratch

## Overview
Binary classification using a Decision Tree and a Random Forest, both
implemented with only NumPy. No sklearn used for the models themselves
sklearn is used solely for benchmarking.

## Math

**Gini Impurity** (per node):
| Component | Formula |
|---|---|
| Gini impurity | Gini = 1 - Σ(p_i²) |
| Weighted Gini (children) | (n_left/n) × Gini(left) + (n_right/n) × Gini(right) |
| Information Gain | Gini(parent) - WeightedGini(children) |

Splits are chosen by scanning every feature and every candidate threshold,
picking whichever maximizes Information Gain.

**Random Forest** adds two randomness sources on top of the tree:
- Bootstrap sampling: each tree trains on `n` rows drawn with replacement
- Random feature subsets: each split considers only `sqrt(n_features)`
  randomly chosen features, forcing trees to diverge structurally

## Dataset
Breast Cancer Wisconsin 569 samples, 30 features.
Target: Malignant (0) vs Benign (1).
No feature scaling applied tree splits are threshold comparisons, so
monotonic transforms like StandardScaler have no effect on the result.

## Results

### Decision Tree (max_depth=4)
| Model | Test Accuracy |
|---|---|
| Custom | 93.86% |
| Sklearn | 93.86% |

Confusion matrices were identical between implementations.

### Random Forest (n_trees=100, max_depth=4)
| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Custom | 99.34% | 94.74% |
| Sklearn | 99.34% | 95.61% |

## Confusion Matrix Decision Tree
```
                      Predicted
                 Malignant  Benign
Actual Malignant      39      3
Actual Benign         4       68
```
