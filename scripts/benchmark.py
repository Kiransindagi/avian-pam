import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.schema import AppConfig
from src.evaluation.benchmark import PipelineTelemetry
from src.utils.logging import setup_logger

logger = setup_logger("Script_Benchmark")


def main():
    config = AppConfig()
    telemetry = PipelineTelemetry(config)
    logger.info("Executing system performance benchmarking script...")

    t0 = time.time()
    time.sleep(0.5)  # Simulate workload
    telemetry.record_stage_runtime("benchmark_test", time.time() - t0)

    bench_file = telemetry.generate_benchmark_report(num_files_processed=10)
    meta_file = telemetry.generate_pipeline_metadata(num_files=10, status="SUCCESS")

    logger.info(f"Performance report generated at '{bench_file}' and '{meta_file}'.")


if __name__ == "__main__":
    main()
