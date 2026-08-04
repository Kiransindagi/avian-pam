from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from src.models.base_model import BaseAvianModel
from src.models.model_registry import register_model


@register_model("svr")
class SVRModel(BaseAvianModel):
    """Support Vector Regression (SVR) model."""

    def __init__(
        self, kernel: str = "rbf", C: float = 1.0, epsilon: float = 0.1, **kwargs
    ):
        super().__init__(name="svr", kernel=kernel, C=C, epsilon=epsilon, **kwargs)
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon, **kwargs)

    @property
    def supported_feature_types(self) -> List[str]:
        return ["dsp", "bioacoustics", "embeddings"]

    def _fit_internal(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


@register_model("knn")
class KNNRegressorModel(BaseAvianModel):
    """K-Nearest Neighbors (KNN) Regressor model."""

    def __init__(self, n_neighbors: int = 5, weights: str = "distance", **kwargs):
        super().__init__(name="knn", n_neighbors=n_neighbors, weights=weights, **kwargs)
        self.model = KNeighborsRegressor(
            n_neighbors=n_neighbors, weights=weights, **kwargs
        )

    @property
    def supported_feature_types(self) -> List[str]:
        return ["dsp", "bioacoustics", "embeddings"]

    def _fit_internal(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
