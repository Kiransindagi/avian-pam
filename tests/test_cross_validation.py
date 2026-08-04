import numpy as np
import pandas as pd

from src.models.baselines import LinearRegressionModel
from src.training.cross_validation import CrossValidationEngine


def test_group_kfold_cross_validation():
    cv = CrossValidationEngine(strategy="group_kfold", n_splits=3)
    X = pd.DataFrame(np.random.randn(15, 4), columns=[f"f{i}" for i in range(4)])
    y = pd.Series(np.random.randint(1, 10, size=15), name="bird_count")
    groups = pd.Series([f"aviary_{i % 3}" for i in range(15)])

    model = LinearRegressionModel()
    results = cv.evaluate_model(model, X, y, groups=groups)

    assert "mean_mae" in results
    assert "mean_rmse" in results
    assert "overall_r2" in results
    assert results["n_splits"] == 3
    assert len(results["oof_predictions"]) == 15
