import numpy as np

from src.config.schema import AppConfig
from src.models.baselines import RidgeRegressionModel
from src.models.model_registry import ModelRegistryManager


def test_model_registry(tmp_path):
    config = AppConfig()
    config.paths.artifacts_dir = str(tmp_path)
    registry_mgr = ModelRegistryManager(config)

    model = RidgeRegressionModel(alpha=1.0)
    X = np.random.randn(10, 3)
    y = np.random.randint(1, 5, size=10)
    model.fit(X, y)

    metrics = {"mean_mae": 1.2, "mean_rmse": 1.5, "overall_r2": 0.5}
    artifacts = registry_mgr.register_model_checkpoint(model, metrics)

    assert artifacts["artifact"].exists()
    assert artifacts["meta"].exists()
