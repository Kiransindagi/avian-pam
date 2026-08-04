import librosa
import numpy as np
from typing import Dict, Any, List
from src.features.base import BaseFeatureExtractor
from src.features.registry import register_extractor


@register_extractor("bioacoustics")
class BioacousticFeatureExtractor(BaseFeatureExtractor):
    """Production Bioacoustic & Ecoacoustic Feature Extractor.

    Extracts biologically meaningful ecoacoustic indices (ACI, BI, Entropy, NDSI)
    and call density statistics derived from bioacoustics literature for bird abundance estimation.
    """

    def __init__(self, n_fft: int = 2048, hop_length: int = 512, **kwargs):
        super().__init__(n_fft=n_fft, hop_length=hop_length, **kwargs)
        self.n_fft = n_fft
        self.hop_length = hop_length

    @property
    def name(self) -> str:
        return "bioacoustics"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def dependencies(self) -> List[str]:
        return ["librosa", "numpy", "scipy"]

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "aci": float,
            "bioacoustic_index": float,
            "acoustic_entropy_h": float,
            "temporal_entropy_ht": float,
            "spectral_entropy_hf": float,
            "ndsi": float,
            "acoustic_occupancy": float,
            "call_density": float,
            "chorus_intensity": float,
            "inter_call_interval_mean": float,
            "inter_call_interval_std": float,
            "call_burstiness": float,
            "soundscape_diversity": float,
        }

    @property
    def feature_dimension(self) -> int:
        return 13

    @property
    def computational_complexity(self) -> str:
        return "O(N log N)"

    def extract(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        if len(y) == 0:
            return {k: 0.0 for k in self.output_schema.keys()}

        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)

        # -------------------------------------------------------------
        # 1. Acoustic Complexity Index (ACI) - Pieretti et al. (2011)
        # Correlates with vocalization complexity & multiple calling birds
        # -------------------------------------------------------------
        dS = np.abs(np.diff(S, axis=1))
        sum_dS = np.sum(dS, axis=1)
        sum_S = np.sum(S, axis=1)
        valid_bins = sum_S > 1e-7
        aci_per_bin = sum_dS[valid_bins] / sum_S[valid_bins]
        aci_val = float(np.sum(aci_per_bin))

        # -------------------------------------------------------------
        # 2. Bioacoustic Index (BI) - Boelman et al. (2007)
        # Area under mean dB spectrum in biophony band (2000 - 8000 Hz)
        # -------------------------------------------------------------
        bio_mask = (freqs >= 2000) & (freqs <= 8000)
        mean_spectrum_db = librosa.amplitude_to_db(np.mean(S, axis=1), ref=np.max)
        if np.any(bio_mask):
            biophony_db = mean_spectrum_db[bio_mask] - np.min(
                mean_spectrum_db[bio_mask]
            )
            bioacoustic_index = float(np.trapz(biophony_db))
        else:
            bioacoustic_index = 0.0

        # -------------------------------------------------------------
        # 3. Acoustic Entropy Index (H = Ht * Hf) - Sueur et al. (2008)
        # -------------------------------------------------------------
        # Temporal entropy Ht
        rms_env = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        rms_norm = rms_env / max(1e-7, np.sum(rms_env))
        rms_norm = rms_norm[rms_norm > 0]
        ht = (
            float(-np.sum(rms_norm * np.log2(rms_norm)) / np.log2(len(rms_norm)))
            if len(rms_norm) > 1
            else 0.0
        )

        # Spectral entropy Hf
        spec_mean = np.mean(S, axis=0)
        spec_norm = spec_mean / max(1e-7, np.sum(spec_mean))
        spec_norm = spec_norm[spec_norm > 0]
        hf = (
            float(-np.sum(spec_norm * np.log2(spec_norm)) / np.log2(len(spec_norm)))
            if len(spec_norm) > 1
            else 0.0
        )

        acoustic_entropy_h = float(ht * hf)

        # -------------------------------------------------------------
        # 4. Normalized Difference Soundscape Index (NDSI) - Kasten et al. (2012)
        # (Biophony 2-8kHz - Anthrophony 1-2kHz) / (Biophony + Anthrophony)
        # -------------------------------------------------------------
        anthro_mask = (freqs >= 1000) & (freqs < 2000)
        biophony_power = np.sum(S[bio_mask, :] ** 2) if np.any(bio_mask) else 0.0
        anthrophony_power = (
            np.sum(S[anthro_mask, :] ** 2) if np.any(anthro_mask) else 0.0
        )

        ndsi_denom = biophony_power + anthrophony_power
        ndsi = float((biophony_power - anthrophony_power) / max(1e-7, ndsi_denom))

        # -------------------------------------------------------------
        # 5. Acoustic Occupancy & Call Density
        # -------------------------------------------------------------
        frame_duration = self.hop_length / sr
        noise_floor = np.median(rms_env) + np.std(rms_env)
        active_frames = rms_env > noise_floor
        acoustic_occupancy = float(np.sum(active_frames) / max(1, len(active_frames)))

        # Onset detect for call density & inter-call intervals
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=self.hop_length
        )
        onset_times = onset_frames * frame_duration
        duration_sec = len(y) / sr
        call_density = float(len(onset_times) / max(0.001, duration_sec))

        if len(onset_times) > 1:
            ici = np.diff(onset_times)
            ici_mean = float(np.mean(ici))
            ici_std = float(np.std(ici))
            call_burstiness = float(
                (ici_std - ici_mean) / max(1e-7, ici_std + ici_mean)
            )
        else:
            ici_mean, ici_std, call_burstiness = 0.0, 0.0, 0.0

        # Chorus Intensity (mean dB in biophony band during active frames)
        if np.any(active_frames) and np.any(bio_mask):
            chorus_intensity = float(
                np.mean(
                    librosa.amplitude_to_db(
                        S[bio_mask, :][:, active_frames], ref=np.max
                    )
                )
            )
        else:
            chorus_intensity = 0.0

        # Soundscape Diversity (Shannon entropy across frequency channels)
        freq_dist = np.sum(S, axis=1)
        freq_norm = freq_dist / max(1e-7, np.sum(freq_dist))
        freq_norm = freq_norm[freq_norm > 0]
        soundscape_diversity = (
            float(-np.sum(freq_norm * np.log2(freq_norm)))
            if len(freq_norm) > 0
            else 0.0
        )

        return {
            "aci": aci_val,
            "bioacoustic_index": bioacoustic_index,
            "acoustic_entropy_h": acoustic_entropy_h,
            "temporal_entropy_ht": ht,
            "spectral_entropy_hf": hf,
            "ndsi": ndsi,
            "acoustic_occupancy": acoustic_occupancy,
            "call_density": call_density,
            "chorus_intensity": chorus_intensity,
            "inter_call_interval_mean": ici_mean,
            "inter_call_interval_std": ici_std,
            "call_burstiness": call_burstiness,
            "soundscape_diversity": soundscape_diversity,
        }
