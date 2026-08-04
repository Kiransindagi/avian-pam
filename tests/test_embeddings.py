import numpy as np

from src.features.embeddings import (
    BirdNETEmbedder,
    EmbeddingCacheManager,
    PANNsEmbedder,
)


def test_pretrained_embeddings(tmp_path):
    cache_mgr = EmbeddingCacheManager(cache_dir=tmp_path)
    sr = 32000
    y = (np.random.randn(sr) * 0.1).astype(np.float32)

    birdnet = BirdNETEmbedder(dimension=16)
    emb = birdnet.embed(y, sr)
    assert emb.shape == (16,)

    panns = PANNsEmbedder(dimension=32)
    p_emb = panns.embed(y, sr)
    assert p_emb.shape == (32,)

    # Test cache put/get
    cache_mgr.put("birdnet", y, sr, emb)
    cached_emb = cache_mgr.get("birdnet", y, sr)
    assert cached_emb is not None
    assert np.allclose(emb, cached_emb)
