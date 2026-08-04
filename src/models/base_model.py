import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd


class BaseAvianModel(ABC):
    """Abstract Base Class for all Bird Population Estimation ML Models.

    Provides standardized fit, predict, save, load, and introspection methods.
    """

    def __init__(self, name: str, version: str = "1.0.0", **kwargs):
        self._name = name
        self._version = version
        self.kwargs = kwargs
        self.is_fitted = False
        self.training_time_sec: float = 0.0
        self.inference_latency_ms: float = 0.0
        self.feature_names: List[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    @abstractmethod
    def supported_feature_types(self) -> List[str]:
        """Returns list of supported feature categories (e.g. ['dsp', 'bioacoustics', 'embeddings'])."""
        pass

    @abstractmethod
    def _fit_internal(self, X: np.ndarray, y: np.ndarray):
        """Internal fit implementation for specific estimator algorithm."""
        pass

    @abstractmethod
    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        """Internal predict implementation for specific estimator algorithm."""
        pass

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> "BaseAvianModel":
        """Fits model on training features and target count values."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)

        y_arr = np.asarray(y, dtype=np.float32)

        t0 = time.time()
        self._fit_internal(X_arr, y_arr)
        self.training_time_sec = float(time.time() - t0)
        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predicts estimated bird counts for given feature array."""
        if not self.is_fitted:
            raise RuntimeError(
                f"Model '{self.name}' must be fitted before calling predict()."
            )

        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)

        t0 = time.time()
        preds = self._predict_internal(X_arr)
        latency = (time.time() - t0) * 1000.0 / max(1, len(X_arr))
        self.inference_latency_ms = float(latency)

        # Enforce non-negative bird count estimates
        return np.maximum(0.0, preds)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        """Optional probability or confidence estimation (returns None for regressors)."""
        return None

    def get_parameters(self) -> Dict[str, Any]:
        """Returns model hyperparameters and metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "is_fitted": self.is_fitted,
            "training_time_sec": self.training_time_sec,
            "inference_latency_ms": self.inference_latency_ms,
            "hyperparameters": self.kwargs,
            "feature_names": self.feature_names,
        }

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Returns dictionary of feature importance scores if supported by estimator."""
        return None

    def save(self, file_path: Path) -> Path:
        """Serializes model instance to disk checkpoint."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)
        return file_path

    @classmethod
    def load(cls, file_path: Path) -> "BaseAvianModel":
        """Deserializes model instance from disk checkpoint."""
        model = joblib.load(Path(file_path))
        if not isinstance(model, BaseAvianModel):
            raise TypeError(
                f"Loaded object from '{file_path}' is not a BaseAvianModel instance."
            )
        return model
