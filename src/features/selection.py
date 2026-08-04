import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Type, Optional
from src.utils.logging import setup_logger

logger = setup_logger("FeatureSelection")

_SELECTOR_REGISTRY: Dict[str, Type["BaseFeatureSelector"]] = {}


def register_selector(name: str):
    def decorator(cls: Type["BaseFeatureSelector"]):
        _SELECTOR_REGISTRY[name] = cls
        return cls

    return decorator


class BaseFeatureSelector(ABC):
    """Abstract Base Class for Feature Selection Modules."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def select_features(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = "bird_count",
    ) -> List[str]:
        """Returns list of selected feature column names."""
        pass


@register_selector("variance_threshold")
class VarianceThresholdSelector(BaseFeatureSelector):
    """Selects features exceeding a minimum variance threshold."""

    def __init__(self, threshold: float = 1e-4, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.threshold = threshold

    def select_features(
        self, df: pd.DataFrame, target_col: Optional[str] = "bird_count"
    ) -> List[str]:
        non_feat_cols = ["file_path", "filename", "species", target_col]
        feature_cols = [
            c
            for c in df.columns
            if c not in non_feat_cols and np.issubdtype(df[c].dtype, np.number)
        ]

        variances = df[feature_cols].var()
        selected = list(variances[variances >= self.threshold].index)
        logger.info(
            f"VarianceThresholdSelector: Retained {len(selected)} / {len(feature_cols)} features (threshold={self.threshold})."
        )
        return selected


@register_selector("correlation_filter")
class CorrelationFilterSelector(BaseFeatureSelector):
    """Removes collinear features with absolute correlation above threshold."""

    def __init__(self, threshold: float = 0.90, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.threshold = threshold

    def select_features(
        self, df: pd.DataFrame, target_col: Optional[str] = "bird_count"
    ) -> List[str]:
        non_feat_cols = ["file_path", "filename", "species", target_col]
        feature_cols = [
            c
            for c in df.columns
            if c not in non_feat_cols and np.issubdtype(df[c].dtype, np.number)
        ]

        corr_matrix = df[feature_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [
            column for column in upper.columns if any(upper[column] > self.threshold)
        ]

        selected = [c for c in feature_cols if c not in to_drop]
        logger.info(
            f"CorrelationFilterSelector: Retained {len(selected)} / {len(feature_cols)} features (threshold={self.threshold})."
        )
        return selected


@register_selector("tree_importance")
class TreeImportanceSelector(BaseFeatureSelector):
    """Selects top K features using a RandomForest / ExtraTrees surrogate estimator."""

    def __init__(self, top_k: int = 15, **kwargs):
        super().__init__(top_k=top_k, **kwargs)
        self.top_k = top_k

    def select_features(
        self, df: pd.DataFrame, target_col: Optional[str] = "bird_count"
    ) -> List[str]:
        non_feat_cols = ["file_path", "filename", "species", target_col]
        feature_cols = [
            c
            for c in df.columns
            if c not in non_feat_cols and np.issubdtype(df[c].dtype, np.number)
        ]

        if target_col not in df.columns or df[target_col].isnull().all():
            logger.warning(
                "Target column missing for TreeImportanceSelector. Falling back to all features."
            )
            return feature_cols

        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)

        try:
            from sklearn.ensemble import RandomForestRegressor

            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)
            importances = pd.Series(rf.feature_importances_, index=feature_cols)
            selected = list(
                importances.sort_values(ascending=False).head(self.top_k).index
            )
            logger.info(
                f"TreeImportanceSelector: Selected top {len(selected)} features by Random Forest importance."
            )
            return selected
        except Exception as e:
            logger.warning(
                f"TreeImportanceSelector failed: {e}. Returning all features."
            )
            return feature_cols


def get_feature_selector(name: str, **kwargs) -> BaseFeatureSelector:
    """Instantiates a feature selector plugin by name."""
    if name not in _SELECTOR_REGISTRY:
        raise KeyError(
            f"Feature selector '{name}' not found. Available: {list(_SELECTOR_REGISTRY.keys())}"
        )
    return _SELECTOR_REGISTRY[name](**kwargs)
