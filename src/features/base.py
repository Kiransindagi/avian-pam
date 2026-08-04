from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np


class BaseFeatureExtractor(ABC):
    """Abstract Base Class for all Acoustic Feature Extractors in Avian PAM Platform.

    All extractors must implement this interface to support automatic registration,
    metadata introspection, and config-driven pipeline execution.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the extractor plugin."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Semantic version of the feature extractor plugin (e.g. '1.0.0')."""
        pass

    @property
    @abstractmethod
    def dependencies(self) -> List[str]:
        """List of required Python libraries (e.g. ['librosa', 'scipy', 'numpy'])."""
        pass

    @property
    @abstractmethod
    def output_schema(self) -> Dict[str, Any]:
        """Mapping of feature names to data types (e.g. {'rms_mean': float})."""
        pass

    @property
    @abstractmethod
    def feature_dimension(self) -> int:
        """Total number of scalar feature dimensions output by this extractor."""
        pass

    @property
    @abstractmethod
    def computational_complexity(self) -> str:
        """Big-O complexity representation (e.g. 'O(N log N)', 'O(N)')."""
        pass

    @property
    def configurable_parameters(self) -> Dict[str, Any]:
        """Dictionary of user-configurable extraction parameters."""
        return self.kwargs

    @abstractmethod
    def extract(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extracts acoustic features from an audio time-series array.

        Args:
            y: Audio time-series numpy array (1D float32)
            sr: Sampling rate in Hz

        Returns:
            Dictionary mapping feature names to scalar numerical values.
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Returns introspective metadata dictionary for feature provenance."""
        return {
            "name": self.name,
            "version": self.version,
            "dependencies": self.dependencies,
            "feature_dimension": self.feature_dimension,
            "computational_complexity": self.computational_complexity,
            "parameters": self.configurable_parameters,
        }
