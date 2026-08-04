import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.config.schema import AppConfig
from src.utils.io import compute_file_hash, ensure_dir
from src.utils.logging import setup_logger

logger = setup_logger("FeatureStore")


class FeatureStore:
    """Enterprise Feature Store supporting Normalization, Versioning, Schema Metadata, Parquet/CSV Export, and Provenance."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.store_dir = ensure_dir(config.paths.feature_store_dir)
        self.format = getattr(config.features, "store_format", "parquet").lower()

    def normalize_features(
        self,
        df: pd.DataFrame,
        method: str = "standard",
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        """Applies normalization (standard, minmax, robust) to numerical feature columns."""
        non_feat_cols = ["file_path", "filename", "species", "bird_count"]
        feature_cols = [
            c
            for c in df.columns
            if c not in non_feat_cols and np.issubdtype(df[c].dtype, np.number)
        ]

        if not feature_cols:
            return df.copy(), {}

        scaled_df = df.copy()
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(
                f"Scaling method '{method}' not supported. Allowed: standard, minmax, robust."
            )

        scaled_values = scaler.fit_transform(scaled_df[feature_cols].fillna(0))
        scaled_df[feature_cols] = scaled_values

        stats_meta = {}
        for col in feature_cols:
            stats_meta[col] = {
                "mean": float(scaled_df[col].mean()),
                "std": float(scaled_df[col].std()),
                "min": float(scaled_df[col].min()),
                "max": float(scaled_df[col].max()),
            }

        return scaled_df, stats_meta

    def save_features(
        self,
        features_df: pd.DataFrame,
        version: str = "v1.0.0",
        dataset_name: str = "avian_features",
    ) -> Tuple[Path, Path]:
        """Saves feature dataframe and metadata JSON sidecar into feature store."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{dataset_name}_{version}_{timestamp}"

        if self.format == "parquet":
            data_file = self.store_dir / f"{filename_base}.parquet"
            features_df.to_parquet(data_file, index=False)
        else:
            data_file = self.store_dir / f"{filename_base}.csv"
            features_df.to_csv(data_file, index=False)

        schema_metadata = {
            "dataset_name": dataset_name,
            "version": version,
            "created_at": timestamp,
            "num_rows": len(features_df),
            "num_features": len(features_df.columns),
            "columns": list(features_df.columns),
            "dtypes": {col: str(dtype) for col, dtype in features_df.dtypes.items()},
            "storage_format": self.format,
            "data_file_hash": compute_file_hash(data_file),
            "project": self.config.project.name,
        }

        meta_file = self.store_dir / f"{filename_base}_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(schema_metadata, f, indent=2)

        logger.info(
            f"Feature store saved dataset to '{data_file}' ({len(features_df)} rows)."
        )
        return data_file, meta_file

    def save_feature_bundle(
        self,
        df: pd.DataFrame,
        dataset_name: str = "biodcase_avian_features",
        version: str = "2.0.0",
        normalization_method: Optional[str] = "standard",
        extractor_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """Exports raw, normalized, and statistical feature artifacts to feature store."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        raw_filename = f"{dataset_name}_raw_v{version}_{timestamp}.parquet"
        raw_path = self.store_dir / raw_filename
        df.to_parquet(raw_path, index=False)
        raw_hash = compute_file_hash(raw_path)

        norm_df, norm_stats = self.normalize_features(
            df, method=normalization_method or "standard"
        )
        norm_filename = (
            f"{dataset_name}_norm_{normalization_method}_v{version}_{timestamp}.parquet"
        )
        norm_path = self.store_dir / norm_filename
        norm_df.to_parquet(norm_path, index=False)
        norm_hash = compute_file_hash(norm_path)

        non_feat_cols = ["file_path", "filename", "species", "bird_count"]
        feature_cols = [c for c in df.columns if c not in non_feat_cols]

        provenance_metadata = {
            "dataset_name": dataset_name,
            "version": version,
            "timestamp": timestamp,
            "environment": self.config.project.environment,
            "num_records": len(df),
            "num_features": len(feature_cols),
            "feature_names": feature_cols,
            "raw_artifact": {"filename": raw_filename, "file_hash": raw_hash},
            "normalized_artifact": {
                "filename": norm_filename,
                "file_hash": norm_hash,
                "normalization_method": normalization_method,
            },
            "extractor_metadata": extractor_metadata or {},
        }

        meta_path = (
            self.store_dir / f"{dataset_name}_v{version}_{timestamp}_metadata.json"
        )
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(provenance_metadata, f, indent=2)

        stats_path = (
            self.store_dir / f"{dataset_name}_v{version}_{timestamp}_statistics.json"
        )
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(norm_stats, f, indent=2)

        logger.info(
            f"Feature Store: Saved raw ({raw_filename}) and normalized ({norm_filename}) bundles."
        )

        return {
            "raw_parquet": raw_path,
            "norm_parquet": norm_path,
            "metadata_json": meta_path,
            "statistics_json": stats_path,
        }

    def load_latest_features(
        self, dataset_name: str = "avian_features"
    ) -> Optional[pd.DataFrame]:
        """Loads most recent feature dataset matching dataset_name."""
        files = list(self.store_dir.glob(f"{dataset_name}_*.*"))
        data_files = [f for f in files if f.suffix in [".parquet", ".csv"]]

        if not data_files:
            logger.warning(f"No existing feature datasets found for '{dataset_name}'.")
            return None

        latest_file = max(data_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"Loading latest feature store file: '{latest_file}'")

        if latest_file.suffix == ".parquet":
            return pd.read_parquet(latest_file)
        else:
            return pd.read_csv(latest_file)
