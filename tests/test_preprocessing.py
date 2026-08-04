import numpy as np
import pytest

from src.config.schema import AppConfig
from src.data.preprocessing import AudioPreprocessor


@pytest.fixture
def preprocessor():
    cfg = AppConfig()
    return AudioPreprocessor(cfg)


def test_to_mono(preprocessor):
    stereo = np.ones((2, 1000))
    mono = preprocessor.to_mono(stereo)
    assert mono.ndim == 1
    assert len(mono) == 1000


def test_normalize_peak(preprocessor):
    signal = np.array([0.1, -0.5, 0.2, -0.4], dtype=np.float32)
    norm = preprocessor.normalize(signal)
    max_amp = np.max(np.abs(norm))
    target_amp = 10 ** (preprocessor.config.target_peak_db / 20.0)
    assert np.isclose(max_amp, target_amp, atol=1e-3)
