import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from src.config.schema import AppConfig
from src.models.base_model import BaseAvianModel
from src.utils.io import compute_file_hash, ensure_dir
from src.utils.logging import setup_logger

logger = setup_logger("ModelRegistryManager")

_MODEL_REGISTRY: Dict[str, Type[BaseAvianModel]] = {}


def register_model(name: str):
    """Decorator for registering ML model classes into global factory."""

    def decorator(cls: Type[BaseAvianModel]):
        if name in _MODEL_REGISTRY:
            logger.warning(f"Overwriting registered model '{name}'")
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str, **kwargs) -> BaseAvianModel:
    """Instantiates registered model class by name."""
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Model '{name}' not found in registry. "
            f"Available models: {list_registered_models()}"
        )
    return _MODEL_REGISTRY[name](**kwargs)


def list_registered_models() -> List[str]:
    """Returns list of registered model identifiers."""
    return list(_MODEL_REGISTRY.keys())


class ModelRegistryManager:
    """Enterprise Model Registry for trained model checkpoints and sidecar metadata."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.models_artifact_dir = ensure_dir(Path("models"))
        self.checkpoints_dir = ensure_dir(Path("checkpoints"))
        self.registry_file = self.models_artifact_dir / "model_registry.json"

    def register_model_checkpoint(
        self,
        model: BaseAvianModel,
        metrics: Dict[str, Any],
        feature_set_name: str = "biodcase_avian_features",
        dataset_version: str = "2.0.0",
    ) -> Dict[str, Path]:
        """Saves model binary checkpoint and updates model_registry.json catalog."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model.name}_v{model.version}_{timestamp}.joblib"

        artifact_model_path = self.models_artifact_dir / model_filename
        checkpoint_path = self.checkpoints_dir / model_filename

        # Save model binary
        model.save(artifact_model_path)
        model.save(checkpoint_path)
        file_hash = compute_file_hash(artifact_model_path)

        metadata = {
            "model_name": model.name,
            "version": model.version,
            "timestamp": timestamp,
            "environment": self.config.project.environment,
            "dataset_version": dataset_version,
            "feature_set": feature_set_name,
            "training_time_sec": model.training_time_sec,
            "inference_latency_ms": model.inference_latency_ms,
            "hyperparameters": model.get_parameters().get("hyperparameters", {}),
            "metrics": metrics,
            "artifact_path": str(artifact_model_path),
            "checkpoint_path": str(checkpoint_path),
            "file_hash": file_hash,
        }

        registry_data = self._load_registry()
        registry_data.append(metadata)

        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(registry_data, f, indent=2)

        meta_path = (
            self.models_artifact_dir
            / f"{model.name}_v{model.version}_{timestamp}_meta.json"
        )
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"ModelRegistry: Registered model '{model.name}' at '{artifact_model_path}'."
        )
        return {
            "artifact": artifact_model_path,
            "checkpoint": checkpoint_path,
            "meta": meta_path,
        }

    def _load_registry(self) -> List[Dict[str, Any]]:
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load model_registry.json: {e}")
                return []
        return []
