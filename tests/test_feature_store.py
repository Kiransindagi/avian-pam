import pytest
import pandas as pd
from src.config.schema import AppConfig
from src.features.feature_store import FeatureStore


def test_feature_store_save_and_load(tmp_path):
    cfg = AppConfig()
    cfg.paths.feature_store_dir = tmp_path / "feature_store"
    cfg.features.store_format = "csv"

    store = FeatureStore(cfg)

    df = pd.DataFrame(
        {
            "filename": ["rec1.wav", "rec2.wav"],
            "bird_count": [3, 5],
            "rms_mean": [0.05, 0.08],
        }
    )

    data_file, meta_file = store.save_features(df, version="v1.0.0", dataset_name="test_dataset")

    assert data_file.exists()
    assert meta_file.exists()

    loaded_df = store.load_latest_features("test_dataset")
    assert loaded_df is not None
    assert len(loaded_df) == 2
    assert "rms_mean" in loaded_df.columns
