# KNN From Scratch

Implementation of K-Nearest Neighbors classifier using only NumPy, compared against sklearn on the Wine dataset.

## Overview

No ML libraries used for the model itself. The goal is to understand the algorithm mechanically, from distance calculation to majority voting, and verify correctness by matching sklearn's output.

## Dataset

Wine dataset (sklearn built-in) 178 samples, 13 numerical features, 3 classes (wine cultivars). Chosen because all features are numerical (Euclidean distance applies directly), features have wildly different scales (makes scaling impact dramatic and measurable), and 3 classes shows KNN generalizes beyond binary classification.

## Pipeline

```
EDA -> Outlier Capping (99th percentile) -> Stratified Split -> StandardScaler -> KNN From Scratch -> Evaluation
```

## Implementation

```python
class KNNClassifier:
    def fit(self, X, y):
        self.fit_X = X   # no computation lazy learner
        self.fit_y = y

    def euclidean_distance(self, x):
        return np.sqrt(((self.fit_X - x) ** 2).sum(axis=1))  # vectorized, no loop

    def predict_single(self, x):
        distances = self.euclidean_distance(x)
        k_indices = np.argsort(distances)[:self.k]
        k_labels  = self.fit_y[k_indices]
        return np.bincount(k_labels).argmax()

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])
```

Key design decisions:
- Distance computation fully vectorized using NumPy broadcasting no Python loop inside `euclidean_distance`
- `fit()` does zero computation all cost is deferred to prediction time (lazy learner)
- Majority vote via `np.bincount().argmax()`

## Results

| | My KNN | sklearn KNN |
|---|---|---|
| K used | 7 | 5 |
| Accuracy | 1.0000 | 0.9722 |

Both implementations use identical logic the difference is K value only.

## Effect of Scaling

| | Accuracy |
|---|---|
| Without StandardScaler | 0.7222 |
| With StandardScaler | 1.0000 |

A 27.78% accuracy gap from one preprocessing step. `proline` (range 400-1600) and `magnesium` (range 70-160) dominated Euclidean distances completely, making the model ignore the remaining 11 features. Scaling brings every feature to mean=0, std=1 and fixes this.

## K Tuning

Best K found at K=7 via accuracy vs K plot across K=1 to 20. Even-numbered K values (2, 4) performed worse due to tie-breaking issues in majority voting across 3 classes.

## Key Concepts Demonstrated

- Why KNN is a lazy learner no training computation, all cost at inference
- Why scaling is non-negotiable for distance-based models
- Vectorized distance computation with NumPy broadcasting
- Stratified train-test split for multi-class problems

## Stack

Python, NumPy, pandas, matplotlib, seaborn, scikit-learn (for dataset, scaling, and comparison only)
