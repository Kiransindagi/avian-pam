from abc import ABC, abstractmethod
import numpy as np


class BaseAudioEmbedder(ABC):
    """Abstract Base Class for Pretrained Deep Audio Embedding Extractors."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier of the pretrained embedding model (e.g. 'birdnet', 'panns')."""
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Dimensionality of vector representation output by this model."""
        pass

    @abstractmethod
    def embed(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Computes dense feature embedding array for input audio signal.

        Args:
            y: Audio waveform numpy array (1D float32)
            sr: Sample rate in Hz

        Returns:
            1D numpy array of shape (embedding_dimension,)
        """
        pass
