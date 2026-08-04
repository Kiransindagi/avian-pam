import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict
import pandas as pd
from src.config.schema import AppConfig
from src.utils.io import compute_file_hash, ensure_dir
from src.utils.logging import setup_logger

logger = setup_logger("DatasetVersioning")


class DatasetVersionManager:
    """Manages semantic dataset versioning, manifest hashes, and statistical metadata sidecars."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.data_dir = Path(config.paths.raw_data_dir).parent
        self.versions_dir = ensure_dir(self.data_dir / "versions")
        self.checksums_dir = ensure_dir(self.data_dir / "checksums")
        self.metadata_dir = ensure_dir(self.data_dir / "metadata")

    def create_version_manifest(
        self,
        version: str = "v1.0.0",
        description: str = "BioDCASE2026 Raw & Preprocessed Dataset Release",
    ) -> Dict[str, Path]:
        """Generates version.json, hash.json, and statistics.json for the dataset."""
        raw_dir = Path(self.config.paths.raw_data_dir)
        processed_dir = Path(self.config.paths.processed_data_dir)

        all_files = list(raw_dir.rglob("*.wav")) + list(raw_dir.rglob("*.flac"))
        file_hashes: Dict[str, str] = {}
        file_sizes = []

        for f in all_files:
            rel_name = str(f.relative_to(raw_dir))
            f_hash = compute_file_hash(f, algorithm="md5")
            file_hashes[rel_name] = f_hash
            file_sizes.append(f.stat().st_size)

        # Build Manifest Hash
        manifest_string = json.dumps(file_hashes, sort_keys=True)
        dataset_hash = hashlib.sha256(manifest_string.encode("utf-8")).hexdigest()

        # 1. version.json
        version_data = {
            "dataset_name": self.config.project.name,
            "version": version,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "dataset_hash": dataset_hash,
            "num_raw_files": len(all_files),
            "num_processed_files": len(list(processed_dir.rglob("*.wav")))
            if processed_dir.exists()
            else 0,
            "environment": self.config.project.environment,
        }
        version_path = self.versions_dir / "version.json"
        with open(version_path, "w", encoding="utf-8") as f:
            json.dump(version_data, f, indent=2)

        # 2. hash.json
        hash_data = {
            "dataset_hash": dataset_hash,
            "hash_algorithm": "sha256_manifest",
            "file_hashes": file_hashes,
        }
        hash_path = self.versions_dir / "hash.json"
        with open(hash_path, "w", encoding="utf-8") as f:
            json.dump(hash_data, f, indent=2)

        # 3. statistics.json
        stats_data = {
            "total_files": len(all_files),
            "total_size_bytes": sum(file_sizes),
            "avg_file_size_bytes": float(pd.Series(file_sizes).mean())
            if file_sizes
            else 0.0,
            "target_sample_rate": self.config.audio.target_sample_rate,
            "target_channels": self.config.audio.target_channels,
        }
        meta_csv = raw_dir / self.config.paths.metadata_filename
        if meta_csv.exists():
            try:
                meta_df = pd.read_csv(meta_csv)
                stats_data["num_metadata_records"] = len(meta_df)
                if "bird_count" in meta_df.columns:
                    stats_data["mean_bird_count"] = float(meta_df["bird_count"].mean())
                    stats_data["max_bird_count"] = int(meta_df["bird_count"].max())
                if "species" in meta_df.columns:
                    stats_data["num_unique_species"] = int(meta_df["species"].nunique())
            except Exception as e:
                logger.warning(f"Could not parse metadata stats: {e}")

        stats_path = self.versions_dir / "statistics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, indent=2)

        logger.info(
            f"Dataset versioning manifest created successfully: version '{version}' (hash: {dataset_hash[:10]}...)"
        )
        return {
            "version.json": version_path,
            "hash.json": hash_path,
            "statistics.json": stats_path,
        }
