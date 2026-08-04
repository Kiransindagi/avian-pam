from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeRegressor

from src.models.base_model import BaseAvianModel
from src.models.model_registry import register_model
from src.utils.logging import setup_logger

logger = setup_logger("TreeModels")


@register_model("decision_tree")
class DecisionTreeModel(BaseAvianModel):
    """Decision Tree Regressor model."""

    def __init__(self, max_depth: Optional[int] = 10, random_state: int = 42, **kwargs):
        super().__init__(
            name="decision_tree",
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )
        self.model = DecisionTreeRegressor(
            max_depth=max_depth, random_state=random_state, **kwargs
        )

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
            col: float(imp)
            for col, imp in zip(self.feature_names, self.model.feature_importances_)
        }


@register_model("random_forest")
class RandomForestModel(BaseAvianModel):
    """Random Forest Regressor ensemble model."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 12,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            name="random_forest",
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            **kwargs,
        )

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
            col: float(imp)
            for col, imp in zip(self.feature_names, self.model.feature_importances_)
        }


@register_model("extra_trees")
class ExtraTreesModel(BaseAvianModel):
    """Extremely Randomized Trees Regressor model."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 12,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            name="extra_trees",
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )
        self.model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            **kwargs,
        )

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
            col: float(imp)
            for col, imp in zip(self.feature_names, self.model.feature_importances_)
        }


@register_model("gradient_boosting")
class GradientBoostingModel(BaseAvianModel):
    """Scikit-Learn Gradient Boosting Regressor model."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            name="gradient_boosting",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )

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
            col: float(imp)
            for col, imp in zip(self.feature_names, self.model.feature_importances_)
        }


@register_model("hist_gradient_boosting")
class HistGradientBoostingModel(BaseAvianModel):
    """Histogram-based Gradient Boosting Regressor model."""

    def __init__(
        self,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            name="hist_gradient_boosting",
            max_iter=max_iter,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs,
        )
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs,
        )

    @property
    def supported_feature_types(self) -> List[str]:
        return ["dsp", "bioacoustics", "embeddings"]

    def _fit_internal(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


@register_model("xgboost")
class XGBoostModel(BaseAvianModel):
    """XGBoost Regressor model with fallback to GradientBoostingRegressor."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            name="xgboost",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )
        try:
            import xgboost as xgb

            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
                **kwargs,
            )
            self._using_fallback = False
        except ImportError:
            logger.warning(
                "XGBoost library not installed. Falling back to GradientBoostingRegressor."
            )
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
            )
            self._using_fallback = True

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
            col: float(imp)
            for col, imp in zip(self.feature_names, self.model.feature_importances_)
        }


@register_model("lightgbm")
class LightGBMModel(BaseAvianModel):
    """LightGBM Regressor model with fallback to HistGradientBoostingRegressor."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            name="lightgbm",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )
        try:
            import lightgbm as lgb

            self.model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
                verbose=-1,
                **kwargs,
            )
            self._using_fallback = False
        except ImportError:
            logger.warning(
                "LightGBM library not installed. Falling back to HistGradientBoostingRegressor."
            )
            self.model = HistGradientBoostingRegressor(
                max_iter=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
            )
            self._using_fallback = True

    @property
    def supported_feature_types(self) -> List[str]:
        return ["dsp", "bioacoustics", "embeddings"]

    def _fit_internal(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


@register_model("catboost")
class CatBoostModel(BaseAvianModel):
    """CatBoost Regressor model with fallback to GradientBoostingRegressor."""

    def __init__(
        self,
        iterations: int = 100,
        learning_rate: float = 0.1,
        depth: int = 5,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            name="catboost",
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_state=random_state,
            **kwargs,
        )
        try:
            from catboost import CatBoostRegressor

            self.model = CatBoostRegressor(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                random_seed=random_state,
                verbose=0,
                **kwargs,
            )
            self._using_fallback = False
        except ImportError:
            logger.warning(
                "CatBoost library not installed. Falling back to GradientBoostingRegressor."
            )
            self.model = GradientBoostingRegressor(
                n_estimators=iterations,
                learning_rate=learning_rate,
                max_depth=depth,
                random_state=random_state,
            )
            self._using_fallback = True

    @property
    def supported_feature_types(self) -> List[str]:
        return ["dsp", "bioacoustics", "embeddings"]

    def _fit_internal(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
