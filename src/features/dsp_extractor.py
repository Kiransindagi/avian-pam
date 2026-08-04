import librosa
import numpy as np
import scipy.stats as stats
from typing import Dict, Any, List
from src.features.base import BaseFeatureExtractor
from src.features.registry import register_extractor


@register_extractor("dsp")
class DSPFeatureExtractor(BaseFeatureExtractor):
    """Production Digital Signal Processing (DSP) Acoustic Feature Extractor.
    
    Extracts time domain, frequency domain, cepstral, mel-scale, pitch, and temporal
    envelope features from bioacoustic recordings for avian population estimation.
    """

    def __init__(
        self,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 64,
        **kwargs,
    ):
        super().__init__(
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            **kwargs,
        )
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    @property
    def name(self) -> str:
        return "dsp"

    @property
    def version(self) -> str:
        return "2.0.0"

    @property
    def dependencies(self) -> List[str]:
        return ["librosa", "numpy", "scipy"]

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "rms_mean": float, "rms_std": float, "peak_amplitude": float,
            "crest_factor": float, "dynamic_range_db": float, "signal_entropy": float,
            "zcr_mean": float, "zcr_std": float, "spectral_centroid_mean": float,
            "spectral_centroid_std": float, "spectral_bandwidth_mean": float,
            "spectral_bandwidth_std": float, "spectral_contrast_mean": float,
            "spectral_contrast_std": float, "spectral_flatness_mean": float,
            "spectral_flatness_std": float, "spectral_rolloff_mean": float,
            "spectral_rolloff_std": float, "spectral_flux_mean": float,
            "harmonic_ratio": float, "silence_ratio": float, "activity_ratio": float,
            "onset_rate": float, "energy_skewness": float, "energy_kurtosis": float,
            "f0_mean": float, "f0_std": float, "pitch_stability": float,
        }

    @property
    def feature_dimension(self) -> int:
        # Base features (28) + MFCCs (n_mfcc * 2) + Delta MFCCs (n_mfcc * 2) + Delta-Delta MFCCs (n_mfcc * 2) + Mel Stats (4)
        return 28 + (self.n_mfcc * 6) + 4

    @property
    def computational_complexity(self) -> str:
        return "O(N log N)"

    def extract(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        features: Dict[str, float] = {}

        if len(y) == 0:
            return {k: 0.0 for k in self.output_schema.keys()}

        # -------------------------------------------------------------
        # 1. Time Domain Features
        # -------------------------------------------------------------
        rms_frame = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        rms_mean = float(np.mean(rms_frame))
        rms_std = float(np.std(rms_frame))
        peak_amp = float(np.max(np.abs(y)))
        crest_factor = float(peak_amp / max(1e-7, rms_mean))

        min_rms = float(np.min(rms_frame[rms_frame > 1e-7])) if np.any(rms_frame > 1e-7) else 1e-7
        dynamic_range_db = float(20.0 * np.log10(peak_amp / min_rms))

        # Signal Entropy
        abs_y = np.abs(y)
        sum_y = np.sum(abs_y)
        if sum_y > 0:
            prob = abs_y / sum_y
            prob = prob[prob > 0]
            signal_entropy = float(-np.sum(prob * np.log2(prob)))
        else:
            signal_entropy = 0.0

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=self.hop_length)[0]

        features.update({
            "rms_mean": rms_mean,
            "rms_std": rms_std,
            "peak_amplitude": peak_amp,
            "crest_factor": crest_factor,
            "dynamic_range_db": dynamic_range_db,
            "signal_entropy": signal_entropy,
            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr)),
        })

        # -------------------------------------------------------------
        # 2. Frequency Domain Features
        # -------------------------------------------------------------
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)[0]
        band = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)[0]
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=self.n_fft, hop_length=self.hop_length)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)[0]

        # Spectral Flux
        stft_mag = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        flux = np.sqrt(np.sum(np.diff(stft_mag, axis=1) ** 2, axis=0)) if stft_mag.shape[1] > 1 else np.array([0.0])

        # Harmonic Ratio
        y_harm, y_perc = librosa.effects.hpss(y)
        harm_energy = np.sum(y_harm ** 2)
        total_energy = np.sum(y ** 2)
        harmonic_ratio = float(harm_energy / max(1e-7, total_energy))

        features.update({
            "spectral_centroid_mean": float(np.mean(cent)),
            "spectral_centroid_std": float(np.std(cent)),
            "spectral_bandwidth_mean": float(np.mean(band)),
            "spectral_bandwidth_std": float(np.std(band)),
            "spectral_contrast_mean": float(np.mean(contrast)),
            "spectral_contrast_std": float(np.std(contrast)),
            "spectral_flatness_mean": float(np.mean(flatness)),
            "spectral_flatness_std": float(np.std(flatness)),
            "spectral_rolloff_mean": float(np.mean(rolloff)),
            "spectral_rolloff_std": float(np.std(rolloff)),
            "spectral_flux_mean": float(np.mean(flux)),
            "harmonic_ratio": harmonic_ratio,
        })

        # -------------------------------------------------------------
        # 3. Cepstral Features (MFCC, Delta, Delta-Delta)
        # -------------------------------------------------------------
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        for i in range(self.n_mfcc):
            features[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i]))
            features[f"mfcc_{i+1}_std"] = float(np.std(mfccs[i]))
            features[f"mfcc_delta_{i+1}_mean"] = float(np.mean(delta_mfccs[i]))
            features[f"mfcc_delta_{i+1}_std"] = float(np.std(delta_mfccs[i]))
            features[f"mfcc_delta2_{i+1}_mean"] = float(np.mean(delta2_mfccs[i]))
            features[f"mfcc_delta2_{i+1}_std"] = float(np.std(delta2_mfccs[i]))

        # -------------------------------------------------------------
        # 4. Mel Spectrogram Statistics
        # -------------------------------------------------------------
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        features["mel_spec_mean"] = float(np.mean(mel_db))
        features["mel_spec_std"] = float(np.std(mel_db))
        features["mel_spec_skew"] = float(stats.skew(mel_db.flatten()))
        features["mel_spec_kurtosis"] = float(stats.kurtosis(mel_db.flatten()))

        # -------------------------------------------------------------
        # 5. Pitch & Fundamental Frequency Features
        # -------------------------------------------------------------
        try:
            f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
            valid_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
            if len(valid_f0) > 0:
                f0_mean = float(np.mean(valid_f0))
                f0_std = float(np.std(valid_f0))
            else:
                f0_mean, f0_std = 0.0, 0.0
        except Exception:
            f0_mean, f0_std = 0.0, 0.0

        features["f0_mean"] = f0_mean
        features["f0_std"] = f0_std
        features["pitch_stability"] = float(1.0 / (1.0 + f0_std))

        # -------------------------------------------------------------
        # 6. Temporal & Energy Envelope Features
        # -------------------------------------------------------------
        silence_threshold = 0.01 * np.max(rms_frame) if len(rms_frame) > 0 else 0.01
        silent_frames = np.sum(rms_frame < silence_threshold)
        silence_ratio = float(silent_frames / max(1, len(rms_frame)))
        activity_ratio = float(1.0 - silence_ratio)

        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=self.hop_length)
        duration_sec = len(y) / sr
        onset_rate = float(len(onsets) / max(0.001, duration_sec))

        features["silence_ratio"] = silence_ratio
        features["activity_ratio"] = activity_ratio
        features["onset_rate"] = onset_rate
        features["energy_skewness"] = float(stats.skew(rms_frame)) if len(rms_frame) > 2 else 0.0
        features["energy_kurtosis"] = float(stats.kurtosis(rms_frame)) if len(rms_frame) > 2 else 0.0

        return features
