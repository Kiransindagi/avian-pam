from typing import Dict, Type, List, Any
from src.features.base import BaseFeatureExtractor
from src.utils.logging import setup_logger

logger = setup_logger("FeatureRegistry")

_EXTRACTOR_REGISTRY: Dict[str, Type[BaseFeatureExtractor]] = {}


def register_extractor(name: str):
    """Decorator to register a new feature extractor plugin.

    Example:
        @register_extractor("bioacoustics")
        class BioacousticExtractor(BaseFeatureExtractor):
            ...
    """

    def decorator(cls: Type[BaseFeatureExtractor]):
        if name in _EXTRACTOR_REGISTRY:
            logger.warning(f"Overwriting registered feature extractor plugin '{name}'")
        _EXTRACTOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_extractor(name: str, **kwargs) -> BaseFeatureExtractor:
    """Instantiates a registered feature extractor plugin by name."""
    if name not in _EXTRACTOR_REGISTRY:
        raise KeyError(
            f"Feature extractor '{name}' not found in registry. "
            f"Available extractors: {list_registered_extractors()}"
        )
    return _EXTRACTOR_REGISTRY[name](**kwargs)


def list_registered_extractors() -> List[str]:
    """Returns a list of all currently registered feature extractor plugin names."""
    return list(_EXTRACTOR_REGISTRY.keys())


def get_registry_metadata() -> Dict[str, Dict[str, Any]]:
    """Returns metadata across all registered feature extractors."""
    meta = {}
    for name, cls in _EXTRACTOR_REGISTRY.items():
        try:
            inst = cls()
            meta[name] = inst.get_metadata()
        except Exception:
            meta[name] = {"name": name, "class": cls.__name__}
    return meta
