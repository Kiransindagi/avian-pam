from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

from src.config.schema import AppConfig
from src.utils.logging import setup_logger

logger = setup_logger("DummyGenerator")


def generate_dummy_dataset(config: AppConfig, n_samples: int = 10) -> Path:
    """Generate synthetic bioacoustic audio files and metadata.csv for testing."""
    raw_dir = config.paths.raw_data_dir
    target_dir = raw_dir / "dev_aviary_1" / "chunk_000"
    target_dir.mkdir(parents=True, exist_ok=True)

    sr = config.audio.target_sample_rate
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    metadata_records = []

    for i in range(n_samples):
        filename = f"rec_d1_00_00_{i+1:02d}.000000.wav"
        filepath = target_dir / filename

        # Generate synthetic audio with background noise + chirp signals
        noise = np.random.normal(0, 0.01, len(t))
        chirp = 0.2 * np.sin(2 * np.pi * (1000 + 1500 * (i + 1) * t) * t)
        audio_signal = noise + chirp
        audio_signal = audio_signal / np.max(np.abs(audio_signal))

        sf.write(filepath, audio_signal, sr)

        bird_count = (i % 5) + 1
        metadata_records.append(
            {
                "file_path": str(filepath.relative_to(raw_dir)),
                "filename": filename,
                "duration_sec": duration,
                "sample_rate": sr,
                "bird_count": bird_count,
                "species": "avian_mix",
            }
        )

    meta_df = pd.DataFrame(metadata_records)
    meta_path = raw_dir / config.paths.metadata_filename
    meta_df.to_csv(meta_path, index=False)

    logger.info(
        f"Generated {n_samples} dummy audio recordings and metadata at '{raw_dir}'"
    )
    return raw_dir
