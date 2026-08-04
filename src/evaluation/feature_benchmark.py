import json
import os
import time
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from src.config.schema import AppConfig
from src.features.base import BaseFeatureExtractor
from src.utils.io import ensure_dir
from src.utils.logging import setup_logger

logger = setup_logger("FeatureBenchmark")


class FeatureBenchmarkSuite:
    """Enterprise Performance & Latency Benchmark Engine for Feature Extractors."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.reports_dir = ensure_dir(config.paths.reports_dir)
        self.process = psutil.Process(os.getpid())

    def benchmark_extractor(
        self,
        extractor: BaseFeatureExtractor,
        y_test: np.ndarray,
        sr: int = 32000,
        n_repeats: int = 3,
    ) -> Dict[str, Any]:
        """Measures extraction speed, memory RSS delta, and feature dimensions for a plugin."""
        runtimes = []
        mem_before = self.process.memory_info().rss / (1024 * 1024)

        for _ in range(n_repeats):
            t0 = time.time()
            feats = extractor.extract(y_test, sr)
            t1 = time.time()
            runtimes.append(t1 - t0)

        mem_after = self.process.memory_info().rss / (1024 * 1024)
        avg_runtime_sec = float(np.mean(runtimes))
        audio_duration_sec = len(y_test) / sr
        realtime_factor = float(avg_runtime_sec / max(0.001, audio_duration_sec))
        throughput_samples_sec = float(len(y_test) / max(0.0001, avg_runtime_sec))

        return {
            "extractor_name": extractor.name,
            "version": extractor.version,
            "feature_dimension": extractor.feature_dimension,
            "avg_extraction_sec": round(avg_runtime_sec, 5),
            "realtime_factor": round(realtime_factor, 4),
            "throughput_samples_sec": round(throughput_samples_sec, 2),
            "memory_rss_mb_delta": round(max(0.0, mem_after - mem_before), 2),
            "computational_complexity": extractor.computational_complexity,
        }

    def run_suite_and_export(
        self,
        extractors: List[BaseFeatureExtractor],
        sample_audio_len_sec: float = 5.0,
        sr: int = 32000,
    ) -> Path:
        """Runs benchmark benchmark suite across all extractors and exports feature_benchmark.json."""
        rng = np.random.RandomState(42)
        y_test = (rng.randn(int(sample_audio_len_sec * sr)) * 0.1).astype(np.float32)

        results = []
        for ext in extractors:
            try:
                res = self.benchmark_extractor(ext, y_test, sr=sr)
                results.append(res)
            except Exception as e:
                logger.error(f"Failed benchmarking extractor '{ext.name}': {e}")

        benchmark_report = {
            "project": self.config.project.name,
            "environment": self.config.project.environment,
            "num_extractors_benchmarked": len(results),
            "sample_audio_duration_sec": sample_audio_len_sec,
            "sample_rate_hz": sr,
            "extractor_benchmarks": results,
        }

        out_path = self.reports_dir / "feature_benchmark.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_report, f, indent=2)

        logger.info(f"Saved feature benchmark report to '{out_path}'.")
        return out_path
