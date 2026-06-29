# Logistic Regression from Scratch

## Overview
Binary classification using Logistic Regression implemented 
with only NumPy. No sklearn used for the model itself.

## Math
| Component | Formula |
|-----------|---------|
| Linear combination | z = X·W + b |
| Sigmoid | σ(z) = 1 / (1 + e^(-z)) |
| Loss (BCE) | L = -(1/n) × Σ[y·log(ŷ) + (1-y)·log(1-ŷ)] |
| Gradient W | dW = (1/n) × Xᵀ·(ŷ - y) |
| Gradient b | db = mean(ŷ - y) |
| Update | W = W - α·dW |

## Dataset
Breast Cancer Wisconsin 569 samples, 30 features.
Target: Malignant (0) vs Benign (1).

## Results
| Iterations | Accuracy | Malignant Recall |
|------------|----------|-----------------|
| 1,000 | 95.6% | 0.95 |
| 10,000 | 97.4% | 0.98 |
| Sklearn | 98.2% | 0.98 |

## Key Insight
Recall on malignant class is the critical metric missing a 
malignant tumor is far more costly than a false alarm.
Gap with sklearn comes from optimizer difference: 
gradient descent vs LBFGS.
