# Linear Regression from Scratch

## What
Implementing Linear Regression using gradient descent with only NumPy.
No sklearn used for the model itself only for verification and data loading.

## Math
- Prediction: ŷ = X·W + b
- Loss (MSE): L = (1/n) × Σ(ŷ - y)²
- Gradient W: dW = (1/n) × Xᵀ·(ŷ - y)
- Gradient b: db = mean(ŷ - y)
- Update: W = W - α·dW

## Results
| Metric | My Implementation | Sklearn |
|--------|------------------|---------|
| MSE | 0.4845 | 0.4625 |
| R² | 0.630 | 0.647 |

## Dataset
California Housing 20,640 samples, 8 features, predicting median house value.

## Key learnings
- Gradient descent converges but doesn't reach the exact optimal solution
- Sklearn uses closed form (normal equation) which is exact but O(n³)
- Gradient descent scales better to large datasets
