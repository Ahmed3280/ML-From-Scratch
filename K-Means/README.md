# K-Means Clustering from Scratch

Implemented K-Means clustering using only NumPy on the
Mall Customer Segmentation dataset.

## Dataset
**Mall Customer Segmentation** 200 customers, 2 features:
Annual Income (k$) and Spending Score (1–100).

## Pipeline
| Step | Details |
|------|---------|
| EDA | Distributions, scatter plot 5 clusters visible pre-fitting |
| Preprocessing | Drop ID, select 2 features, StandardScaler on full dataset |
| Elbow Method | WCSS for K=1..10 elbow at K=5 |
| K-Means Scratch | NumPy only random init, Euclidean distance, convergence check |
| Sklearn Compare | Same K, same data verify our implementation |
| Evaluation | Silhouette Score, Davies-Bouldin Index, Inertia |

## Results

| Metric | Our K-Means | Sklearn |
|--------|-------------|---------|
| Inertia (WCSS) | 65.5789 | 65.5684 |
| Silhouette Score | 0.5539 | 0.5547 |
| Davies-Bouldin Index | 0.5708 | 0.5722 |

Converged at iteration 7.
Note: our random init with random_state=42 happened to find the global
optimum in one run. On harder datasets, multiple restarts (n_init > 1)
or k-means++ initialization would be necessary.

## Key concepts implemented
- 3D broadcasting for vectorized distance computation:
  `(n,1,d) - (1,k,d) → (n,k,d) → norm(axis=2) → (n,k)`
- Boolean masking for cluster assignment: `X[labels == k]`
- Convergence check via centroid shift norm
- Centroid history tracking for visualization

## Why this matters
K-Means is the foundation of unsupervised learning.
Understanding it mechanically why scaling is mandatory,
what WCSS measures, why centroids converge is what
separates someone who runs `KMeans().fit(X)` from someone
who understands what that line actually does.


## Stack
Python · NumPy · Matplotlib · Scikit-learn (comparison only)
