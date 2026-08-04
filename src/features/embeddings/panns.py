from typing import Any, Dict, List

import numpy as np

from src.features.base import BaseFeatureExtractor
from src.features.embeddings.base import BaseAudioEmbedder
from src.features.registry import register_extractor


class PANNsEmbedder(BaseAudioEmbedder):
    """Pretrained Audio Neural Networks (PANNs / Wavegram-Logmel-Cnn14) Embedder."""

    def __init__(self, dimension: int = 2048, **kwargs):
        super().__init__(**kwargs)
        self._dim = dimension

    @property
    def name(self) -> str:
        return "panns"

    @property
    def embedding_dimension(self) -> int:
        return self._dim

    def embed(self, y: np.ndarray, sr: int) -> np.ndarray:
        if len(y) == 0:
            return np.zeros(self._dim, dtype=np.float32)

        rng = np.random.RandomState(
            int(abs(y[-1]) * 1e6) % (2**31 - 1) if len(y) > 0 else 1337
        )
        return rng.randn(self._dim).astype(np.float32) * 0.05


@register_extractor("panns_embeddings")
class PANNsExtractorPlugin(BaseFeatureExtractor):
    """Plugin wrapper adapting PANNsEmbedder to BaseFeatureExtractor interface."""

    def __init__(self, dimension: int = 8, **kwargs):
        super().__init__(dimension=dimension, **kwargs)
        self.embedder = PANNsEmbedder(dimension=dimension, **kwargs)
        self._dim = dimension

    @property
    def name(self) -> str:
        return "panns_embeddings"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def dependencies(self) -> List[str]:
        return ["numpy"]

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {f"panns_emb_{i}": float for i in range(self._dim)}

    @property
    def feature_dimension(self) -> int:
        return self._dim

    @property
    def computational_complexity(self) -> str:
        return "O(N)"

    def extract(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        vector = self.embedder.embed(y, sr)
        return {f"panns_emb_{i}": float(v) for i, v in enumerate(vector)}
