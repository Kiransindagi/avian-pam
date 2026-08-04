from __future__ import annotations

from pathlib import Path
from typing import List, Union

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "BioDCASE2026_Avian_Counting"
    version: str = "1.0.0"
    seed: int = 42
    environment: str = "development"


class PathsConfig(BaseModel):
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    intermediate_dir: Path = Path("data/intermediate")
    feature_store_dir: Path = Path("data/feature_store")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    figures_dir: Path = Path("reports/figures")
    logs_dir: Path = Path("logs")
    metadata_filename: str = "metadata.csv"


class AudioConfig(BaseModel):
    target_sample_rate: int = 32000
    target_channels: int = 1
    target_duration_sec: float = 5.0
    min_duration_sec: float = 0.5
    max_duration_sec: float = 300.0
    valid_extensions: List[str] = [".wav", ".flac", ".ogg", ".mp3"]


class PreprocessingConfig(BaseModel):
    resample: bool = True
    normalize_audio: bool = True
    normalization_type: str = "peak"
    target_peak_db: float = -3.0
    target_rms_db: float = -20.0
    trim_silence: bool = True
    silence_threshold_db: float = -40.0
    remove_noise: bool = False
    filter_cutoff_hz: int = 500


class ValidationConfig(BaseModel):
    strict_mode: bool = False
    hash_algorithm: str = "md5"
    report_file: Path = Path("reports/validation_report.csv")
    log_file: Path = Path("reports/validation.log")


class EDAConfig(BaseModel):
    generate_figures: bool = True
    top_n_species: int = 10
    dpi: int = 300


class FeaturesConfig(BaseModel):
    active_extractors: List[str] = [
        "dsp",
        "bioacoustics",
        "birdnet_embeddings",
        "panns_embeddings",
    ]
    n_mfcc: int = 13
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 64
    store_format: str = "parquet"
    normalization_method: str = "standard"
    feature_selector: str = "correlation_filter"
    variance_threshold: float = 0.0001
    correlation_threshold: float = 0.85


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    eda: EDAConfig = Field(default_factory=EDAConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)

    @classmethod
    def load_from_yaml(cls, yaml_path: Union[str, Path]) -> AppConfig:
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
