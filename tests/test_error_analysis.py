import numpy as np
import pandas as pd

from src.config.schema import AppConfig
from src.evaluation.error_analysis import ErrorAnalyzer


def test_error_analyzer(tmp_path):
    config = AppConfig()
    config.paths.reports_dir = str(tmp_path)
    analyzer = ErrorAnalyzer(config)

    df_feats = pd.DataFrame(
        {
            "filename": [f"f{i}.wav" for i in range(10)],
            "species": [f"sp_{i%2}" for i in range(10)],
            "rms_mean": np.random.randn(10),
        }
    )

    y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y_pred = np.array([1.1, 2.2, 3.1, 4.5, 4.8, 6.2, 7.1, 8.9, 8.8, 10.2])

    res = analyzer.analyze_errors(df_feats, y_true, y_pred, model_name="TestModel")

    assert res["overall_mae"] < 0.5
    assert len(res["worst_predictions"]) <= 5
    assert len(res["best_predictions"]) <= 5
