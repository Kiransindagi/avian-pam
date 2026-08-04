import time
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union

from src.config.schema import AppConfig
from src.data.preprocessing import AudioPreprocessor
from src.features.registry import get_extractor
from src.models.base_model import BaseAvianModel
from src.models.baselines import LinearRegressionModel
from src.utils.logging import setup_logger

logger = setup_logger("ProductionInferenceEngine")


class AvianInferenceEngine:
    """Enterprise Production Inference Engine for Avian Count Estimation."""

    def __init__(self, config: AppConfig, model_checkpoint_path: Optional[Path] = None):
        self.config = config
        self.preprocessor = AudioPreprocessor(config)

        # Active Extractor Plugins
        self.extractors = []
        for name in config.features.active_extractors:
            try:
                ext_inst = get_extractor(
                    name,
                    n_mfcc=config.features.n_mfcc,
                    n_fft=config.features.n_fft,
                    hop_length=config.features.hop_length,
                )
                self.extractors.append(ext_inst)
            except KeyError:
                pass

        # Load Model Checkpoint
        if model_checkpoint_path and Path(model_checkpoint_path).exists():
            logger.info(
                f"InferenceEngine: Loading model checkpoint from '{model_checkpoint_path}'..."
            )
            self.model = BaseAvianModel.load(Path(model_checkpoint_path))
        else:
            logger.info("InferenceEngine: Initializing default fallback model...")
            self.model = LinearRegressionModel()
            # Build dummy feature vector from extractors to set proper feature names
            dummy_y = np.random.randn(16000)
            dummy_feats = {}
            for ext in self.extractors:
                dummy_feats.update(ext.extract(dummy_y, 16000))
            df_dummy = pd.DataFrame([dummy_feats] * 5)
            y_dummy = pd.Series([1, 2, 3, 4, 5])
            self.model.fit(df_dummy, y_dummy)

    def predict_audio_file(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        """Predicts bird count for a single raw audio file."""
        t0 = time.time()
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file '{audio_path}' not found.")

        # Load and preprocess audio
        y, sr = librosa.load(audio_path, sr=self.config.audio.target_sample_rate)
        duration_sec = float(len(y) / sr)

        # Extract features across registered extractor plugins
        features = {}
        for ext in self.extractors:
            feats = ext.extract(y, sr)
            features.update(feats)

        # Build feature DataFrame aligned with model expected inputs
        df_feat = pd.DataFrame([features])
        if self.model.feature_names:
            for col in self.model.feature_names:
                if col not in df_feat.columns:
                    df_feat[col] = 0.0
            df_feat = df_feat[self.model.feature_names]

        # Predict bird count
        pred_count = float(self.model.predict(df_feat)[0])
        latency_ms = (time.time() - t0) * 1000.0

        return {
            "filename": audio_path.name,
            "predicted_bird_count": round(pred_count, 2),
            "estimated_integer_count": int(round(pred_count)),
            "duration_sec": round(duration_sec, 2),
            "inference_latency_ms": round(latency_ms, 2),
            "feature_count_extracted": len(features),
            "model_used": self.model.name,
        }

    def predict_batch_dir(self, audio_dir: Union[str, Path]) -> pd.DataFrame:
        """Runs batch predictions across all audio files in a directory."""
        audio_dir = Path(audio_dir)
        valid_exts = [
            e.lower() if e.startswith(".") else f".{e.lower()}"
            for e in self.config.audio.valid_extensions
        ]
        audio_files = [
            f
            for f in audio_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in valid_exts
        ]

        results = []
        logger.info(
            f"Running batch inference on {len(audio_files)} audio files in '{audio_dir}'..."
        )

        for file_path in audio_files:
            try:
                res = self.predict_audio_file(file_path)
                results.append(res)
            except Exception as e:
                logger.error(f"Inference failed on '{file_path.name}': {e}")

        return pd.DataFrame(results)
