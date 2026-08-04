# Machine Learning Models, Training & Cross-Validation

## Overview
The modeling framework provides an extensible experiment suite comparing 15 predictive algorithms across linear regressors, tree ensembles, kernel models, and stacking meta-learners.

---

## 1. Algorithm Catalog (`src/models/`)

| Family | Model Key | Description |
| :--- | :--- | :--- |
| **Baselines** | `dummy_mean`, `dummy_median` | Statistical benchmark predictors |
| **Linear Models** | `linear_regression`, `ridge`, `lasso`, `elastic_net`, `poisson_regression` | Regularized linear estimators |
| **Tree Ensembles** | `random_forest`, `extra_trees`, `gradient_boosting`, `hist_gradient_boosting`, `xgboost`, `lightgbm`, `catboost` | Non-linear tree boosting & bagging |
| **Kernel / Instance** | `svr`, `knn` | Support Vector Regression & K-Nearest Neighbors |
| **Ensembles** | `voting_ensemble`, `stacking_ensemble` | Multi-model voting and two-stage ridge meta-learning |

---

## 2. Leakage-Free Cross-Validation Scheme (`src/training/cross_validation.py`)

Field acoustic recordings from the same aviary share environmental noise characteristics. Standard random K-Fold CV leads to severe data leakage and artificially optimistic evaluation.

```
       Group K-Fold Split (Grouped by Aviary ID)
┌──────────────────────────────────────────────────────────┐
│ Fold 1: Train on Aviary 2, 3, 4  ──► Evaluate on Aviary 1│
│ Fold 2: Train on Aviary 1, 3, 4  ──► Evaluate on Aviary 2│
│ Fold 3: Train on Aviary 1, 2, 4  ──► Evaluate on Aviary 3│
└──────────────────────────────────────────────────────────┘
```

- **Stratified Group K-Fold**: Guarantees zero overlap of aviary environments between training and validation folds.
- **Metrics Tracked**: Out-of-fold MAE, RMSE, $R^2$, training latency, and inference latency.

---

## 3. Hyperparameter Optimization & Model Registry
- **Random Search Optimizer** (`src/training/optimizer.py`): Performs randomized hyperparameter search across model spaces.
- **Model Registry** (`src/registry/model_registry.py`): Saves fitted model checkpoints (`.joblib`), feature names, hyperparameter metadata, and evaluation metrics under `artifacts/models/`.
