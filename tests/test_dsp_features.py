import numpy as np

from src.features.dsp_extractor import DSPFeatureExtractor


def test_dsp_feature_extraction():
    extractor = DSPFeatureExtractor(n_mfcc=13)
    sr = 32000
    y = (np.random.randn(sr * 2) * 0.1).astype(np.float32)

    features = extractor.extract(y, sr)

    assert isinstance(features, dict)
    assert "rms_mean" in features
    assert "zcr_mean" in features
    assert "spectral_centroid_mean" in features
    assert "mfcc_1_mean" in features
    assert "f0_mean" in features
    assert "silence_ratio" in features
    assert features["rms_mean"] > 0
