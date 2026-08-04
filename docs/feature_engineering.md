# Feature Engineering Platform v2

## Overview
The **Feature Engineering Platform** extracts bioacoustic representations across three complementary acoustic domains using a plugin architecture.

---

## 1. Feature Taxonomy

```
                       Feature Engineering Platform v2
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
┌───────────────┐             ┌───────────────┐             ┌───────────────┐
│  DSP Spectral │             │  Ecoacoustic  │             │ Pretrained    │
│  Features     │             │  Indices      │             │ Embeddings    │
└───────┬───────┘             └───────┬───────┘             └───────┬───────┘
        │                             │                             │
        ├── MFCCs (1-13 + Deltas)     ├── ACI (Complexity)          ├── BirdNET Embeddings
        ├── Spectral Centroid/Flatness├── BI (Bioacoustic Index)    └── PANNs Audio Embeddings
        ├── Zero-Crossing Rate (ZCR)  ├── NDSI (Soundscape Index)
        └── Dynamic Range & Entropy   └── ADI / AEI (Diversity)
```

### A. Digital Signal Processing (DSP) Features
- **Spectral Energy & Envelope**: RMS energy mean/std, peak amplitude, dynamic range (dB), signal entropy.
- **Spectral Shape Descriptors**: Spectral Centroid, Spectral Rolloff, Spectral Bandwidth, Spectral Flatness, Zero-Crossing Rate (ZCR), Harmonic-to-Noise Ratio (HNR).
- **Time-Frequency Representations**: 13 Mel-Frequency Cepstral Coefficients (MFCCs), 13 MFCC Deltas, Mel-spectrogram band energies.

### B. Ecoacoustic Soundscape Indices
- **Acoustic Complexity Index (ACI)**: Measures intensity fluctuations across frequency bands (indicative of biological chorus activity).
- **Bioacoustic Index (BI)**: Quantifies sound level energy within typical avian vocal frequency ranges ($2\text{--}8 \text{ kHz}$).
- **Normalized Difference Soundscape Index (NDSI)**: Measures the ratio of biophony ($2\text{--}8 \text{ kHz}$) to anthrophony ($1\text{--}2 \text{ kHz}$).
- **Acoustic Diversity Index (ADI) & Acoustic Evenness Index (AEI)**: Quantifies signal distribution across frequency channels.

### C. Deep Audio Embeddings
- Pretrained bioacoustic feature representations extracted from deep neural networks trained on broad soundscape classification tasks (BirdNET / PANNs).

---

## 2. Plugin Architecture (`src/features/`)
New extractors are added by inheriting from `BaseFeatureExtractor` and registering via the `@register_extractor` decorator:

```python
from src.features.base import BaseFeatureExtractor
from src.features.registry import register_extractor

@register_extractor("custom_spectral")
class CustomSpectralExtractor(BaseFeatureExtractor):
    def extract(self, y, sr):
        return {"custom_metric": float(np.mean(y**2))}
```

---

## 3. Feature Store v2 & Normalization
Extracted features are stored under `data/feature_store/` with versioned Parquet bundles and integrity checksums:
- **Raw Feature Bundle**: `biodcase_avian_features_raw_v1.0.0.parquet`
- **Normalized Bundle**: Standardized / Robust Scaled `biodcase_avian_features_norm_standard_v1.0.0.parquet`
- **Metadata**: JSON metadata detailing active extractors, timestamp, and feature statistics.
