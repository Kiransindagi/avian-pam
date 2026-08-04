from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class AudioFileContract(BaseModel):
    """Contract enforcing incoming raw audio file specifications."""

    file_path: Path
    file_size_bytes: int = Field(gt=0, description="File size must be greater than zero bytes")
    extension: str

    @field_validator("extension")
    def validate_ext(cls, v):
        allowed = [".wav", ".flac", ".ogg", ".mp3"]
        if v.lower() not in allowed:
            raise ValueError(f"Extension '{v}' violates AudioFileContract. Allowed: {allowed}")
        return v.lower()


class ValidationContract(BaseModel):
    """Contract enforcing data validation output schema."""

    file_path: str
    filename: str
    status: str = Field(pattern="^(VALID|WARNING|CORRUPT|MISSING|DUPLICATE)$")
    error_msg: str = ""
    sample_rate: int = Field(ge=0)
    channels: int = Field(ge=0)
    duration_sec: float = Field(ge=0.0)
    file_hash: str


class PreprocessingContract(BaseModel):
    """Contract enforcing preprocessing output audio integrity."""

    raw_path: Path
    processed_path: Path
    original_sample_rate: int = Field(gt=0)
    target_sample_rate: int = Field(gt=0)
    duration_sec: float = Field(gt=0.0)
    peak_db: float = Field(le=0.0)
    success: bool = True


class FeatureRecordContract(BaseModel):
    """Contract enforcing extracted feature vector schema."""

    file_path: str
    filename: str
    duration_sec: float = Field(gt=0.0)
    sample_rate: int = Field(gt=0)
    bird_count: Optional[int] = Field(default=None, ge=0)
    species: Optional[str] = None
    feature_count: int = Field(gt=0)

    @field_validator("feature_count")
    def validate_feature_count(cls, v):
        if v < 5:
            raise ValueError(f"Feature vector count ({v}) below minimum production contract threshold (5)")
        return v
