import soundfile as sf
import numpy as np
from src.config.schema import AppConfig
from src.data.versioning import DatasetVersionManager


def test_dataset_versioning_manifest(tmp_path):
    cfg = AppConfig()
    cfg.paths.raw_data_dir = tmp_path / "raw"
    cfg.paths.processed_data_dir = tmp_path / "processed"
    cfg.paths.raw_data_dir.mkdir(parents=True, exist_ok=True)

    # Write sample audio
    audio_path = cfg.paths.raw_data_dir / "test_rec.wav"
    sf.write(audio_path, np.sin(np.linspace(0, 1.0, 16000)), 16000)

    vm = DatasetVersionManager(cfg)
    manifest = vm.create_version_manifest(version="v1.0.0-test")

    assert manifest["version.json"].exists()
    assert manifest["hash.json"].exists()
    assert manifest["statistics.json"].exists()
