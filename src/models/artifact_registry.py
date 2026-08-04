import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.config.schema import AppConfig
from src.utils.io import compute_file_hash, ensure_dir
from src.utils.logging import setup_logger

logger = setup_logger("ArtifactRegistry")


class ArtifactRegistry:
    """Enterprise Artifact Registry managing versioned model, data, and pipeline outputs."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.root_dir = ensure_dir(config.paths.artifacts_dir)
        self.categories = [
            "preprocessed_audio",
            "feature_store",
            "validation",
            "eda",
            "logs",
            "models",
            "experiments",
        ]
        self._init_dirs()

    def _init_dirs(self):
        for cat in self.categories:
            ensure_dir(self.root_dir / cat)

    def register_artifact(
        self,
        source_path: Path,
        category: str,
        artifact_name: str,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Copies artifact to versioned registry directory with metadata sidecar."""
        if category not in self.categories:
            raise ValueError(
                f"Category '{category}' not supported. Allowed: {self.categories}"
            )

        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(
                f"Artifact source file '{source_path}' does not exist."
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = ensure_dir(self.root_dir / category / f"v{version}")
        target_file = target_dir / f"{artifact_name}_{timestamp}{source_path.suffix}"

        if source_path.is_dir():
            shutil.copytree(source_path, target_file)
            file_hash = "directory"
        else:
            shutil.copy2(source_path, target_file)
            file_hash = compute_file_hash(target_file)

        artifact_meta = {
            "artifact_name": artifact_name,
            "category": category,
            "version": version,
            "timestamp": timestamp,
            "file_name": target_file.name,
            "file_hash": file_hash,
            "source_path": str(source_path),
            "custom_metadata": metadata or {},
        }

        meta_path = target_dir / f"{artifact_name}_{timestamp}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(artifact_meta, f, indent=2)

        logger.info(
            f"Registered artifact '{artifact_name}' under category '{category}' at '{target_file}'."
        )
        return target_file
