import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.io import ensure_dir
from src.utils.logging import setup_logger

logger = setup_logger("EmbeddingCacheManager")


class EmbeddingCacheManager:
    """Manages disk-backed persistent numpy caching for pretrained audio embeddings."""

    def __init__(self, cache_dir: Path = Path("cache/embeddings")):
        self.cache_dir = ensure_dir(cache_dir)

    def _compute_hash(self, y: np.ndarray, sr: int) -> str:
        """Computes deterministic SHA-256 fingerprint for audio buffer."""
        data_bytes = y.tobytes() + str(sr).encode("utf-8")
        return hashlib.sha256(data_bytes).hexdigest()

    def get(self, model_name: str, y: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Retrieves cached embedding array if present."""
        audio_hash = self._compute_hash(y, sr)
        cache_file = self.cache_dir / f"{model_name}_{audio_hash}.npy"

        if cache_file.exists():
            try:
                emb = np.load(cache_file)
                return emb
            except Exception as e:
                logger.warning(f"Failed loading cache file '{cache_file}': {e}")
                return None
        return None

    def put(self, model_name: str, y: np.ndarray, sr: int, embedding: np.ndarray):
        """Stores calculated embedding array into disk cache."""
        audio_hash = self._compute_hash(y, sr)
        cache_file = self.cache_dir / f"{model_name}_{audio_hash}.npy"
        try:
            np.save(cache_file, embedding)
        except Exception as e:
            logger.warning(f"Failed writing embedding cache to '{cache_file}': {e}")
