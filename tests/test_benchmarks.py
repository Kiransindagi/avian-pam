import pytest
from src.config.schema import AppConfig
from src.evaluation.benchmark import PipelineTelemetry


def test_pipeline_telemetry_reports(tmp_path):
    cfg = AppConfig()
    cfg.paths.reports_dir = tmp_path / "reports"
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)

    telemetry = PipelineTelemetry(cfg)
    telemetry.record_stage_runtime("validation", 0.123)

    bench_path = telemetry.generate_benchmark_report(num_files_processed=5)
    meta_path = telemetry.generate_pipeline_metadata(num_files=5, status="SUCCESS")

    assert bench_path.exists()
    assert meta_path.exists()
