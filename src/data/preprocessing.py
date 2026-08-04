from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf

from src.config.schema import AppConfig
from src.utils.io import ensure_dir
from src.utils.logging import setup_logger

logger = setup_logger("AudioPreprocessor")


class AudioPreprocessor:
    """Modular bioacoustic audio preprocessing pipeline stage."""

    def __init__(self, config: AppConfig):
        self.config = config.preprocessing
        self.audio_cfg = config.audio
        self.paths = config.paths

    def load_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Stage 1: Load audio signal and native sample rate."""
        audio, sr = librosa.load(file_path, sr=None, mono=False)
        return audio, sr

    def to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Stage 2: Convert stereo/multichannel to mono."""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        return audio

    def resample(self, audio: np.ndarray, orig_sr: int) -> Tuple[np.ndarray, int]:
        """Stage 3: Resample audio to target sample rate."""
        target_sr = self.audio_cfg.target_sample_rate
        if self.config.resample and orig_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
            return audio, target_sr
        return audio, orig_sr

    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """Stage 4: Peak or RMS Audio Normalization."""
        if not self.config.normalize_audio or len(audio) == 0:
            return audio

        if self.config.normalization_type == "peak":
            target_amplitude = 10 ** (self.config.target_peak_db / 20.0)
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio * (target_amplitude / max_val)

        elif self.config.normalization_type == "rms":
            target_rms = 10 ** (self.config.target_rms_db / 20.0)
            current_rms = np.sqrt(np.mean(audio**2))
            if current_rms > 0:
                audio = audio * (target_rms / current_rms)

        return audio

    def trim_silence(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Stage 5: Optional silence trimming."""
        if not self.config.trim_silence or len(audio) == 0:
            return audio
        trimmed, _ = librosa.effects.trim(
            audio, top_db=abs(self.config.silence_threshold_db)
        )
        return trimmed if len(trimmed) > 0 else audio

    def process_file(self, raw_file_path: Path, raw_base_dir: Path) -> Path:
        """Executes full preprocessing pipeline on a single file and saves output."""
        audio, orig_sr = self.load_audio(raw_file_path)
        audio = self.to_mono(audio)
        audio, final_sr = self.resample(audio, orig_sr)
        audio = self.normalize(audio)
        audio = self.trim_silence(audio, final_sr)

        # Compute output relative path
        rel_path = raw_file_path.relative_to(raw_base_dir)
        output_file_path = Path(self.paths.processed_data_dir) / rel_path

        ensure_dir(output_file_path.parent)
        sf.write(output_file_path, audio, final_sr)

        return output_file_path

    def process_dataset(self) -> int:
        """Processes all valid raw audio files in dataset."""
        raw_dir = Path(self.paths.raw_data_dir)
        processed_dir = ensure_dir(self.paths.processed_data_dir)

        audio_files = []
        for ext in self.audio_cfg.valid_extensions:
            audio_files.extend(list(raw_dir.rglob(f"*{ext}")))

        logger.info(f"Preprocessing {len(audio_files)} audio files...")
        processed_count = 0

        for f_path in audio_files:
            try:
                self.process_file(f_path, raw_dir)
                processed_count += 1
            except Exception as e:
                logger.error(f"Failed preprocessing for '{f_path.name}': {e}")

        logger.info(
            f"Preprocessing complete. Processed {processed_count}/{len(audio_files)} files to '{processed_dir}'."
        )
        return processed_count
