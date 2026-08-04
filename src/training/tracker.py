import json
import os
import psutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.config.schema import AppConfig
from src.utils.io import ensure_dir
from src.utils.logging import setup_logger

logger = setup_logger("ExperimentTracker")


class ExperimentTracker:
    """Enterprise Experiment Tracking & Governance Engine."""

    def __init__(
        self,
        config: AppConfig,
        registry_file: Path = Path("experiments/experiment_registry.json"),
    ):
        self.config = config
        self.registry_file = registry_file
        self.mlruns_dir = ensure_dir(Path("mlruns"))

    def _get_git_commit(self) -> str:
        try:
            cmd = ["git", "rev-parse", "--short", "HEAD"]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
            return out
        except Exception:
            return "git-unavailable"

    def log_experiment_run(
        self,
        experiment_id: str,
        model_name: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, Any],
        dataset_version: str = "2.0.0",
        feature_version: str = "2.0.0",
    ) -> Path:
        """Logs experiment run metadata and updates experiment_registry.json."""
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Capture hardware telemetry
        process = psutil.Process(os.getpid())
        hardware_info = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
            "cpu_count": psutil.cpu_count(),
        }

        experiment_entry = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "model_name": model_name,
            "environment": self.config.project.environment,
            "dataset_version": dataset_version,
            "feature_version": feature_version,
            "git_commit": self._get_git_commit(),
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "hardware": hardware_info,
        }

        # Update experiment_registry.json
        registry_data = self._load_registry()
        registry_data["registry"].append(experiment_entry)
        registry_data["total_experiments"] = len(registry_data["registry"])

        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(registry_data, f, indent=2)

        # Log MLruns sidecar JSON
        run_file = self.mlruns_dir / f"{experiment_id}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(experiment_entry, f, indent=2)

        logger.info(f"ExperimentTracker: Logged experiment run '{experiment_id}' for model '{model_name}'.")
        return run_file

    def _load_registry(self) -> Dict[str, Any]:
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"project": self.config.project.name, "total_experiments": 0, "registry": []}
