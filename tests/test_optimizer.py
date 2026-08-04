import numpy as np
import pandas as pd

from src.training.optimizer import HyperparameterOptimizer


def test_hyperparameter_optimizer():
    optimizer = HyperparameterOptimizer(n_trials=3)
    X = pd.DataFrame(np.random.randn(12, 4), columns=[f"f{i}" for i in range(4)])
    y = pd.Series(np.random.randint(1, 10, size=12))

    param_dist = {
        "n_estimators": [10, 20],
        "max_depth": [3, 5],
    }

    res = optimizer.optimize_random_search("random_forest", param_dist, X, y)
    assert res["model_name"] == "random_forest"
    assert "best_params" in res
    assert "n_estimators" in res["best_params"]
    assert len(res["trials_history"]) == 3
