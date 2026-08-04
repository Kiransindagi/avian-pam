import pandas as pd

from src.config.schema import AppConfig
from src.features.quality_analyzer import FeatureQualityAnalyzer


def test_quality_analyzer():
    config = AppConfig()
    analyzer = FeatureQualityAnalyzer(config)

    df = pd.DataFrame(
        {
            "file_path": [f"f{i}.wav" for i in range(10)],
            "filename": [f"f{i}.wav" for i in range(10)],
            "rms_mean": [0.1 * i for i in range(10)],
            "rms_std": [0.01 * i for i in range(10)],
            "zero_var": [1.0] * 10,
            "bird_count": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    summary = analyzer.analyze_features(df, target_col="bird_count")
    assert summary["total_features"] == 3
    assert "zero_var" in summary["low_variance_features"]
    assert "rms_mean" in summary["mutual_information"]
