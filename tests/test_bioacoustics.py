import numpy as np
import pytest
from src.features.bioacoustics import BioacousticFeatureExtractor


def test_bioacoustic_indices():
    extractor = BioacousticFeatureExtractor()
    sr = 32000
    y = (np.random.randn(sr * 2) * 0.1).astype(np.float32)

    features = extractor.extract(y, sr)

    assert "aci" in features
    assert "bioacoustic_index" in features
    assert "acoustic_entropy_h" in features
    assert "ndsi" in features
    assert "acoustic_occupancy" in features
    assert "call_density" in features
    assert -1.0 <= features["ndsi"] <= 1.0
