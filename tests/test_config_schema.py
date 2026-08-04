import pytest
from pathlib import Path
from pydantic import ValidationError
from src.config.schemas import (
    AudioFileContract,
    FeatureRecordContract,
)


def test_audio_file_contract_valid():
    contract = AudioFileContract(
        file_path=Path("sample.wav"),
        file_size_bytes=1024,
        extension=".wav",
    )
    assert contract.extension == ".wav"


def test_audio_file_contract_invalid_ext():
    with pytest.raises(ValidationError):
        AudioFileContract(
            file_path=Path("sample.txt"),
            file_size_bytes=1024,
            extension=".txt",
        )


def test_feature_record_contract_valid():
    record = FeatureRecordContract(
        file_path="data/raw/rec1.wav",
        filename="rec1.wav",
        duration_sec=3.0,
        sample_rate=32000,
        feature_count=10,
    )
    assert record.feature_count == 10


def test_feature_record_contract_invalid_count():
    with pytest.raises(ValidationError):
        FeatureRecordContract(
            file_path="data/raw/rec1.wav",
            filename="rec1.wav",
            duration_sec=3.0,
            sample_rate=32000,
            feature_count=2,  # less than minimum 5
        )
