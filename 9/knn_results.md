# KNN on CIFAR-10 with HOG features

Generated 2026-11-06 11:10:28 by `knn.py`.

## Run configuration

| Setting | Value |
| --- | --- |
| Training images | 5000 (500/class) |
| Test images | 1000 (100/class) |
| Classes | 10 |
| Feature dimensions | 1764 |
| Preprocessing | StandardScaler fitted inside each CV fold |
| Cross-validation | Stratified 5-fold, shuffled |
| K values tested | 1, 3, 5, 7, 9, 11, 13, 15, 21, 25, 31 |
| Distance metrics tested | euclidean, manhattan |
| Grid search time | 61.6s |

## Choosing K

Every configuration below was scored on the *same* five folds, so the numbers are directly comparable. The model reported in the next section is the winner refitted on the full training subsample.

### Metric: `euclidean`

| K | Mean accuracy | Std dev | Rank | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 36.18% | 0.99% | 13 | 35.50% | 35.20% | 36.40% | 38.00% | 35.80% |
| 3 | 35.50% | 1.12% | 16 | 35.50% | 33.40% | 35.70% | 36.50% | 36.40% |
| 5 | 36.40% | 1.12% | 12 | 34.70% | 35.50% | 36.90% | 37.20% | 37.70% |
| 7 | 35.70% | 0.98% | 14 | 35.00% | 34.30% | 36.30% | 35.80% | 37.10% |
| 9 | 35.54% | 0.96% | 15 | 34.40% | 35.10% | 36.30% | 34.90% | 37.00% |
| 11 | 35.04% | 1.26% | 17 | 33.30% | 34.50% | 37.00% | 34.60% | 35.80% |
| 13 | 34.38% | 1.70% | 18 | 32.30% | 33.80% | 36.80% | 33.10% | 35.90% |
| 15 | 34.24% | 1.86% | 19 | 31.10% | 34.50% | 36.80% | 33.80% | 35.00% |
| 21 | 33.78% | 1.70% | 20 | 31.20% | 33.00% | 36.40% | 34.40% | 33.90% |
| 25 | 32.94% | 1.79% | 21 | 30.30% | 32.20% | 35.10% | 34.80% | 32.30% |
| 31 | 32.62% | 1.62% | 22 | 30.70% | 32.30% | 34.60% | 34.40% | 31.10% |

Best K for `euclidean`: **5** at 36.40% mean validation accuracy.

### Metric: `manhattan`

| K | Mean accuracy | Std dev | Rank | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 41.70% | 1.43% | 11 | 39.50% | 41.00% | 41.50% | 43.40% | 43.10% |
| 3 | 42.40% | 1.27% | 10 | 42.90% | 41.20% | 40.60% | 43.80% | 43.50% |
| 5 | 45.34% | 0.93% | 2 | 44.20% | 44.50% | 45.20% | 46.30% | 46.50% |
| 7 | 45.36% **(best)** | 0.65% | 1 | 45.30% | 44.60% | 44.80% | 45.70% | 46.40% |
| 9 | 45.14% | 0.60% | 3 | 45.10% | 44.50% | 44.50% | 45.60% | 46.00% |
| 11 | 45.06% | 0.80% | 4 | 45.00% | 44.70% | 43.80% | 46.10% | 45.70% |
| 13 | 44.74% | 1.36% | 5 | 44.40% | 44.30% | 43.20% | 47.30% | 44.50% |
| 15 | 43.78% | 1.78% | 6 | 42.70% | 42.90% | 42.60% | 47.30% | 43.40% |
| 21 | 43.64% | 1.52% | 7 | 41.70% | 42.30% | 43.70% | 45.90% | 44.60% |
| 25 | 43.48% | 1.19% | 8 | 42.60% | 42.00% | 43.60% | 45.50% | 43.70% |
| 31 | 43.30% | 1.42% | 9 | 42.40% | 41.20% | 43.80% | 45.40% | 43.70% |

Best K for `manhattan`: **7** at 45.36% mean validation accuracy.

## Selected model

| Property | Value |
| --- | --- |
| K (n_neighbors) | 7 |
| Distance metric | manhattan |
| Weights | uniform |
| Mean validation accuracy | 45.36% |
| Validation std dev | 0.65% |
| Per-fold validation accuracy | 45.30%, 44.60%, 44.80%, 45.70%, 46.40% |

## Test-set performance (`test_batch`)

| Metric | Value |
| --- | --- |
| Accuracy | 45.40% |
| Precision (macro) | 49.36% |
| Recall (macro) | 45.40% |
| F1-score (macro) | 44.20% |

### Per-class results

| Class | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| airplane | 50.46% | 55.00% | 52.63% | 100 |
| automobile | 64.41% | 76.00% | 69.72% | 100 |
| bird | 43.28% | 29.00% | 34.73% | 100 |
| cat | 27.59% | 8.00% | 12.40% | 100 |
| deer | 26.17% | 39.00% | 31.33% | 100 |
| dog | 52.27% | 23.00% | 31.94% | 100 |
| frog | 30.42% | 80.00% | 44.08% | 100 |
| horse | 64.62% | 42.00% | 50.91% | 100 |
| ship | 59.79% | 58.00% | 58.88% | 100 |
| truck | 74.58% | 44.00% | 55.35% | 100 |

### Confusion matrix

Rows are the true class, columns the predicted class.

| true \ pred | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| airplane | 55 | 1 | 7 | 0 | 8 | 1 | 7 | 1 | 19 | 1 |
| automobile | 4 | 76 | 1 | 0 | 5 | 0 | 6 | 0 | 4 | 4 |
| bird | 8 | 4 | 29 | 3 | 17 | 8 | 28 | 0 | 3 | 0 |
| cat | 4 | 3 | 4 | 8 | 21 | 7 | 37 | 8 | 4 | 4 |
| deer | 3 | 0 | 8 | 4 | 39 | 1 | 40 | 2 | 1 | 2 |
| dog | 2 | 1 | 7 | 11 | 19 | 23 | 31 | 6 | 0 | 0 |
| frog | 3 | 3 | 2 | 0 | 10 | 1 | 80 | 0 | 1 | 0 |
| horse | 4 | 3 | 2 | 2 | 22 | 2 | 16 | 42 | 5 | 2 |
| ship | 15 | 8 | 0 | 1 | 3 | 0 | 10 | 3 | 58 | 2 |
| truck | 11 | 19 | 7 | 0 | 5 | 1 | 8 | 3 | 2 | 44 |
