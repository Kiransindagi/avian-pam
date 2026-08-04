import pandas as pd
import numpy as np
import pytest
from src.features.selection import VarianceThresholdSelector, CorrelationFilterSelector


def test_feature_selection():
    df = pd.DataFrame({
        "file_path": ["1.wav", "2.wav", "3.wav"],
        "f1": [1.0, 2.0, 3.0],
        "f2": [1.0, 2.0, 3.0],  # Collinear with f1
        "f3": [0.5, 0.5, 0.5],  # Low variance
        "bird_count": [1, 2, 3],
    })

    var_selector = VarianceThresholdSelector(threshold=0.01)
    selected_var = var_selector.select_features(df)
    assert "f3" not in selected_var
    assert "f1" in selected_var

    corr_selector = CorrelationFilterSelector(threshold=0.95)
    selected_corr = corr_selector.select_features(df)
    assert len(selected_corr) < 3
