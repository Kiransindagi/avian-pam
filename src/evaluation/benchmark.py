import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import psutil

from src.config.schema import AppConfig
from src.utils.io import ensure_dir
from src.utils.logging import setup_logger

logger = setup_logger("PerformanceTelemetry")


class PipelineTelemetry:
    """Tracks runtime performance metrics, system resource utilization, and exports pipeline_metadata.json."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.reports_dir = ensure_dir(config.paths.reports_dir)
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()
        self.stage_runtimes: Dict[str, float] = {}

    def record_stage_runtime(self, stage_name: str, duration_sec: float):
        """Records execution duration for a specific pipeline stage."""
        self.stage_runtimes[stage_name] = round(duration_sec, 3)
        logger.info(f"Telemetry: Stage '{stage_name}' finished in {duration_sec:.3f}s.")

    def get_system_metrics(self) -> Dict[str, Any]:
        """Collects current system resource utilization."""
        mem_info = self.process.memory_info()
        return {
            "ram_rss_mb": round(mem_info.rss / (1024 * 1024), 2),
            "ram_vms_mb": round(mem_info.vms / (1024 * 1024), 2),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
        }

    def generate_benchmark_report(self, num_files_processed: int) -> Path:
        """Exports benchmark.json containing throughput metrics and resource stats."""
        total_runtime = time.time() - self.start_time
        metrics = self.get_system_metrics()

        benchmark_data = {
            "total_pipeline_runtime_sec": round(total_runtime, 3),
            "stage_runtimes_sec": self.stage_runtimes,
            "throughput_files_per_sec": round(
                num_files_processed / max(0.001, total_runtime), 2
            ),
            "system_resources": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        bench_path = self.reports_dir / "benchmark.json"
        with open(bench_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_data, f, indent=2)

        logger.info(f"Performance benchmark saved to '{bench_path}'.")
        return bench_path

    def generate_pipeline_metadata(
        self,
        dataset_version: str = "v1.0.0",
        num_files: int = 0,
        status: str = "SUCCESS",
    ) -> Path:
        """Exports comprehensive pipeline_metadata.json."""
        git_commit = "uncommitted_local"
        try:
            import subprocess

            git_commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode("utf-8")
                .strip()
            )
        except Exception:
            pass

        pipeline_meta = {
            "pipeline_version": self.config.project.version,
            "project_name": self.config.project.name,
            "environment": self.config.project.environment,
            "execution_time": datetime.now().isoformat(),
            "git_commit": git_commit,
            "dataset_version": dataset_version,
            "number_of_files": num_files,
            "feature_version": self.config.project.version,
            "config_used": {
                "target_sample_rate": self.config.audio.target_sample_rate,
                "active_extractors": self.config.features.active_extractors,
                "normalization_type": self.config.preprocessing.normalization_type,
            },
            "python_version": sys.version.split()[0],
            "os_platform": platform.platform(),
            "execution_status": status,
        }

        meta_path = self.reports_dir / "pipeline_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(pipeline_meta, f, indent=2)

        logger.info(f"Pipeline metadata saved to '{meta_path}'.")
        return meta_path
