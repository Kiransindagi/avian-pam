from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.config.schema import AppConfig
from src.data.validator import AudioValidator


@pytest.fixture
def tmp_config(tmp_path):
    cfg = AppConfig()
    cfg.paths.raw_data_dir = tmp_path / "raw"
    cfg.paths.raw_data_dir.mkdir(parents=True, exist_ok=True)
    cfg.validation.report_file = tmp_path / "reports" / "validation_report.csv"
    cfg.validation.log_file = tmp_path / "reports" / "validation.log"
    return cfg


def test_validator_single_file(tmp_config, tmp_path):
    audio_path = tmp_config.paths.raw_data_dir / "test.wav"
    sr = 32000
    signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr))
    sf.write(audio_path, signal, sr)

    validator = AudioValidator(tmp_config)
    res = validator.validate_single_file(audio_path)

    assert res["status"] == "VALID"
    assert res["sample_rate"] == 32000
    assert res["channels"] == 1
    assert res["duration_sec"] == 1.0


def test_validator_missing_file(tmp_config):
    validator = AudioValidator(tmp_config)
    res = validator.validate_single_file(Path("non_existent_file.wav"))
    assert res["status"] == "MISSING"
