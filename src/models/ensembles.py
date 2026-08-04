from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge

from src.models.base_model import BaseAvianModel
from src.models.model_registry import register_model


@register_model("voting_ensemble")
class VotingEnsembleModel(BaseAvianModel):
    """Voting Regressor ensemble model combining Random Forest, Gradient Boosting, and Ridge."""

    def __init__(self, **kwargs):
        super().__init__(name="voting_ensemble", **kwargs)
        self.rf = RandomForestRegressor(n_estimators=50, random_state=42)
        self.gb = GradientBoostingRegressor(n_estimators=50, random_state=42)
        self.ridge = Ridge(alpha=1.0)
        self.model = VotingRegressor(
            estimators=[("rf", self.rf), ("gb", self.gb), ("ridge", self.ridge)]
        )

    @property
    def supported_feature_types(self) -> List[str]:
        return ["dsp", "bioacoustics", "embeddings"]

    def _fit_internal(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


@register_model("stacking_ensemble")
class StackingEnsembleModel(BaseAvianModel):
    """Stacking Regressor ensemble with Ridge meta-learner."""

    def __init__(self, **kwargs):
        super().__init__(name="stacking_ensemble", **kwargs)
        self.rf = RandomForestRegressor(n_estimators=50, random_state=42)
        self.gb = GradientBoostingRegressor(n_estimators=50, random_state=42)
        self.meta_learner = Ridge(alpha=1.0)
        self.model = StackingRegressor(
            estimators=[("rf", self.rf), ("gb", self.gb)],
            final_estimator=self.meta_learner,
        )

    @property
    def supported_feature_types(self) -> List[str]:
        return ["dsp", "bioacoustics", "embeddings"]

    def _fit_internal(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
