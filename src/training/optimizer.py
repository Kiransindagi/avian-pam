import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.models.model_registry import get_model
from src.training.cross_validation import CrossValidationEngine
from src.utils.logging import setup_logger

logger = setup_logger("HyperparameterOptimizer")


class HyperparameterOptimizer:
    """Hyperparameter Optimization (HPO) Engine for Avian Count Regressors."""

    def __init__(
        self,
        cv_engine: Optional[CrossValidationEngine] = None,
        n_trials: int = 10,
        random_state: int = 42,
    ):
        self.cv_engine = cv_engine or CrossValidationEngine(
            strategy="group_kfold", n_splits=3
        )
        self.n_trials = n_trials
        self.random_state = random_state

    def optimize_random_search(
        self,
        model_name: str,
        param_distributions: Dict[str, List[Any]],
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """Executes Random Search hyperparameter tuning across param_distributions."""
        rng = np.random.RandomState(self.random_state)
        best_mae = float("inf")
        best_params = {}
        trials_history = []

        logger.info(
            f"Starting Random Search HPO for '{model_name}' ({self.n_trials} trials)..."
        )

        for trial_idx in range(self.n_trials):
            # Sample random parameter configuration
            sampled_params = {}
            for param, values in param_distributions.items():
                sampled_params[param] = rng.choice(values)

            try:
                model_inst = get_model(model_name, **sampled_params)
                cv_res = self.cv_engine.evaluate_model(model_inst, X, y, groups=groups)
                score = cv_res["mean_mae"]

                trials_history.append(
                    {
                        "trial": trial_idx + 1,
                        "params": sampled_params,
                        "mae": score,
                        "rmse": cv_res["mean_rmse"],
                    }
                )

                if score < best_mae:
                    best_mae = score
                    best_params = sampled_params

            except Exception as e:
                logger.warning(
                    f"Trial {trial_idx + 1} failed for params {sampled_params}: {e}"
                )

        logger.info(
            f"HPO Complete for '{model_name}'. Best MAE = {best_mae:.4f} with params: {best_params}"
        )

        return {
            "model_name": model_name,
            "best_params": best_params,
            "best_mae": round(best_mae, 4),
            "trials_history": trials_history,
        }
