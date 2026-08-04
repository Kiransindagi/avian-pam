import hashlib
from pathlib import Path
from typing import Union


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """Ensures directory exists and returns Path object."""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_file_hash(file_path: Union[str, Path], algorithm: str = "md5") -> str:
    """Computes file hash for deduplication and validation."""
    hasher = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
