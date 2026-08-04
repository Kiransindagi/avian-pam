import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wave

import numpy as np

from src.config.schema import AppConfig
from src.inference.submission import BioDCASESubmissionGenerator


def create_dummy_wav(path, duration_sec=1.0, sr=22050):
    n_samples = int(duration_sec * sr)
    data = (np.random.randn(n_samples) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())


def test_submission_generator(tmp_path):
    config = AppConfig()
    gen = BioDCASESubmissionGenerator(config)

    audio_file = tmp_path / "eval_sample.wav"
    create_dummy_wav(audio_file)

    sub_path = gen.generate_submission(tmp_path)
    assert sub_path.exists()
    assert sub_path.name.startswith("submission_v")
