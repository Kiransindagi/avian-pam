import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wave

import numpy as np

from src.config.schema import AppConfig
from src.inference.engine import AvianInferenceEngine


def create_dummy_wav(path, duration_sec=1.0, sr=22050):
    n_samples = int(duration_sec * sr)
    data = (np.random.randn(n_samples) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())


def test_inference_engine_single_and_batch(tmp_path):
    config = AppConfig()
    engine = AvianInferenceEngine(config)

    audio_file = tmp_path / "test_bird.wav"
    create_dummy_wav(audio_file)

    # Test single prediction
    res = engine.predict_audio_file(audio_file)
    assert res["filename"] == "test_bird.wav"
    assert "predicted_bird_count" in res
    assert res["inference_latency_ms"] > 0

    # Test batch prediction
    df_batch = engine.predict_batch_dir(tmp_path)
    assert len(df_batch) == 1
    assert "predicted_bird_count" in df_batch.columns
