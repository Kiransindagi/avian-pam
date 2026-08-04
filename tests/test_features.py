import numpy as np

from src.features.dsp_extractor import DSPFeatureExtractor
from src.features.embeddings import BirdNETExtractorPlugin


def test_dsp_extractor():
    extractor = DSPFeatureExtractor()
    y = np.random.randn(32000).astype(np.float32)
    feats = extractor.extract(y, 32000)
    assert "rms_mean" in feats


def test_birdnet_extraction():
    extractor = BirdNETExtractorPlugin(dimension=8)
    y = np.random.randn(32000).astype(np.float32)
    feats = extractor.extract(y, 32000)
    assert len(feats) == 8
    assert "birdnet_emb_0" in feats
