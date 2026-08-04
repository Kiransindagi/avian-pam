from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    PoissonRegressor,
    Ridge,
)

from src.models.base_model import BaseAvianModel
from src.models.model_registry import register_model


@register_model("dummy_mean")
class DummyMeanPredictor(BaseAvianModel):
    """Baseline model predicting constant mean bird count."""

    def __init__(self, **kwargs):
        super().__init__(name="dummy_mean", **kwargs)
        self.model = DummyRegressor(strategy="mean")

    @property
    def supported_feature_types(self) -> List[str]:
        return ["dsp", "bioacoustics", "embeddings"]

    def _fit_internal(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


@register_model("dummy_median")
class DummyMedianPredictor(BaseAvianModel):
    """Baseline model predicting constant median bird count."""

    def __init__(self, **kwargs):
        super().__init__(name="dummy_median", **kwargs)
        self.model = DummyRegressor(strategy="median")

    @property
    def supported_feature_types(self) -> List[str]:
        return ["dsp", "bioacoustics", "embeddings"]

    def _fit_internal(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


@register_model("linear_regression")
class LinearRegressionModel(BaseAvianModel):
    """Ordinary Least Squares Linear Regression baseline."""

    def __init__(self, **kwargs):
        super().__init__(name="linear_regression", **kwargs)
        self.model = LinearRegression(**kwargs)

    @property
    def supported_feature_types(self) -> List[str]:
        return ["dsp", "bioacoustics", "embeddings"]

    def _fit_internal(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self.is_fitted or not self.feature_names:
            return None
        return {
            col: float(coef)
            for col, coef in zip(self.feature_names, np.abs(self.model.coef_))
        }


@register_model("ridge")
class RidgeRegressionModel(BaseAvianModel):
    """L2 Regularized Ridge Regression model."""

    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(name="ridge", alpha=alpha, **kwargs)
        self.model = Ridge(alpha=alpha, **kwargs)

    @property
    def supported_feature_types(self) -> List[str]:
        return ["dsp", "bioacoustics", "embeddings"]

    def _fit_internal(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self.is_fitted or not self.feature_names:
            return None
        return {
            col: float(coef)
            for col, coef in zip(self.feature_names, np.abs(self.model.coef_))
        }


@register_model("lasso")
class LassoRegressionModel(BaseAvianModel):
    """L1 Regularized Lasso Regression model."""

    def __init__(self, alpha: float = 0.1, **kwargs):
        super().__init__(name="lasso", alpha=alpha, **kwargs)
        self.model = Lasso(alpha=alpha, **kwargs)

    @property
    def supported_feature_types(self) -> List[str]:
        return ["dsp", "bioacoustics", "embeddings"]

    def _fit_internal(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self.is_fitted or not self.feature_names:
            return None
        return {
            col: float(coef)
            for col, coef in zip(self.feature_names, np.abs(self.model.coef_))
        }


@register_model("elasticnet")
class ElasticNetModel(BaseAvianModel):
    """L1 + L2 Regularized ElasticNet Regression model."""

    def __init__(self, alpha: float = 0.1, l1_ratio: float = 0.5, **kwargs):
        super().__init__(name="elasticnet", alpha=alpha, l1_ratio=l1_ratio, **kwargs)
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **kwargs)

    @property
    def supported_feature_types(self) -> List[str]:
        return ["dsp", "bioacoustics", "embeddings"]

    def _fit_internal(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


@register_model("poisson")
class PoissonRegressionModel(BaseAvianModel):
    """Poisson GLM model tailored for non-negative count data."""

    def __init__(self, alpha: float = 1.0, max_iter: int = 300, **kwargs):
        super().__init__(name="poisson", alpha=alpha, max_iter=max_iter, **kwargs)
        self.model = PoissonRegressor(alpha=alpha, max_iter=max_iter, **kwargs)

    @property
    def supported_feature_types(self) -> List[str]:
        return ["dsp", "bioacoustics", "embeddings"]

    def _fit_internal(self, X: np.ndarray, y: np.ndarray):
        # Enforce positive count targets for Poisson log link
        y_pos = np.maximum(0.01, y)
        self.model.fit(X, y_pos)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
