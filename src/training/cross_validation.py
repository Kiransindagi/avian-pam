import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Type
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut, RepeatedKFold
from src.models.base_model import BaseAvianModel
from src.evaluation.metrics import compute_avian_metrics
from src.utils.logging import setup_logger

logger = setup_logger("CrossValidationEngine")


class CrossValidationEngine:
    """Enterprise Cross-Validation Strategy Engine tailored for BioDCASE Avian Data."""

    def __init__(
        self,
        strategy: str = "group_kfold",
        n_splits: int = 5,
        n_repeats: int = 1,
        random_state: int = 42,
    ):
        self.strategy = strategy
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def evaluate_model(
        self,
        model: BaseAvianModel,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """Executes leakage-free cross-validation and records out-of-fold metrics."""
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)

        # Fallback groups if none provided
        if groups is None or len(groups) != len(y_arr):
            groups_arr = np.arange(len(y_arr)) % max(2, self.n_splits)
        else:
            groups_arr = groups.to_numpy() if isinstance(groups, pd.Series) else np.asarray(groups)

        num_unique_groups = len(np.unique(groups_arr))
        n_splits = min(self.n_splits, num_unique_groups) if num_unique_groups > 1 else 2

        # Select CV Splitter
        if self.strategy == "group_kfold" and num_unique_groups >= 2:
            cv = GroupKFold(n_splits=n_splits)
            splits = list(cv.split(X_arr, y_arr, groups=groups_arr))
        elif self.strategy == "leave_one_group_out" and num_unique_groups >= 2:
            cv = LeaveOneGroupOut()
            splits = list(cv.split(X_arr, y_arr, groups=groups_arr))
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            splits = list(cv.split(X_arr, y_arr))

        oof_preds = np.zeros(len(y_arr), dtype=np.float32)
        fold_metrics: List[Dict[str, float]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_train, y_train = X_arr[train_idx], y_arr[train_idx]
            X_val, y_val = X_arr[val_idx], y_arr[val_idx]

            # Fit model on training fold
            model.fit(X_train, y_train)
            val_preds = model.predict(X_val)
            oof_preds[val_idx] = val_preds

            fold_res = compute_avian_metrics(y_val, val_preds)
            fold_metrics.append(fold_res)

        # Overall Out-of-Fold Evaluation Metrics
        overall_metrics = compute_avian_metrics(y_arr, oof_preds)
        r2 = float(1.0 - (np.sum((y_arr - oof_preds)**2) / max(1e-7, np.sum((y_arr - np.mean(y_arr))**2))))
        overall_metrics["r2"] = round(r2, 4)

        mean_mae = float(np.mean([m["mae"] for m in fold_metrics]))
        std_mae = float(np.std([m["mae"] for m in fold_metrics]))
        mean_rmse = float(np.mean([m["rmse"] for m in fold_metrics]))
        std_rmse = float(np.std([m["rmse"] for m in fold_metrics]))

        logger.info(
            f"CV [{model.name}] ({self.strategy}): MAE = {mean_mae:.3f} +/- {std_mae:.3f}, "
            f"RMSE = {mean_rmse:.3f} +/- {std_rmse:.3f}, R2 = {r2:.3f}"
        )

        return {
            "model_name": model.name,
            "strategy": self.strategy,
            "n_splits": n_splits,
            "mean_mae": round(mean_mae, 4),
            "std_mae": round(std_mae, 4),
            "mean_rmse": round(mean_rmse, 4),
            "std_rmse": round(std_rmse, 4),
            "overall_r2": round(r2, 4),
            "oof_predictions": oof_preds.tolist(),
            "fold_metrics": fold_metrics,
        }
