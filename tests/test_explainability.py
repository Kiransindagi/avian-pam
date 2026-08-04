import pandas as pd
import numpy as np
import pytest
from src.config.schema import AppConfig
from src.models.trees import RandomForestModel
from src.evaluation.explainability import ExplainabilityEngine


def test_explainability_engine(tmp_path):
    config = AppConfig()
    config.paths.reports_dir = str(tmp_path)
    engine = ExplainabilityEngine(config)

    X = pd.DataFrame(np.random.randn(15, 4), columns=[f"f{i}" for i in range(4)])
    y = pd.Series(np.random.randint(1, 10, size=15))

    model = RandomForestModel(n_estimators=10)
    model.fit(X, y)

    perm_imp = engine.compute_permutation_importance(model, X, y, n_repeats=2)
    assert len(perm_imp) == 4

    shap_vals = engine.compute_shap_surrogate_values(model, X)
    assert shap_vals is not None
    assert shap_vals.shape == (15, 4)
