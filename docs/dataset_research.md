# BioDCASE 2026 Dataset Research & Bioacoustic PAM Analysis Report

## Executive Summary
This report presents an in-depth dataset analysis and engineering risk assessment for the **BioDCASE 2026 Bird Counting Passive Acoustic Monitoring (PAM)** dataset (`Emreargin/BioDCASE2026_Bird_Counting`).

---

## 1. Challenge & Dataset Architecture

### 1.1 Dataset Topology
Passive Acoustic Monitoring (PAM) hardware deployed across aviaries captures continuous bioacoustic soundscapes. The official dataset consists of:
- **Aviaries**: Multi-channel or single-channel autonomous recording units (ARUs) deployed in outdoor aviaries.
- **Audio Chunks**: ~3-second to 5-second acoustic recordings sampled at high resolution (32 kHz / 48 kHz).
- **Labels**: Discrete bird count ground truth (number of simultaneously vocalizing or present birds) along with primary species identifiers.

### 1.2 Folder Hierarchy
```
data/raw/
├── dev_aviary_1/
│   └── chunk_000/
│       ├── rec_d1_00_00_01.000000.wav
│       └── ...
├── dev_aviary_2/
│   └── ...
└── metadata/
    ├── ground_truth.csv
    └── recording_info.csv
```

---

## 2. Statistical Analysis & Key PAM Characteristics

### 2.1 Audio Specifications
- **Sample Rate**: Standardized target at **32,000 Hz** (Nyquist frequency 16 kHz), capturing the full bandwidth of avian vocalizations (1.5 kHz – 8.0 kHz).
- **Channels**: Mono audio streams.
- **Duration**: ~3.0s chunks to capture complete call bouts while keeping window lengths optimal for STFT/Mel feature computation.

### 2.2 Avian Vocalization & Environmental Acoustics
1. **Harmonic Overtones & Formants**: Bird species produce sweeps and chirps with distinct fundamental frequencies ($f_0$) and overtones.
2. **Ambient Noise Profile**: Outdoor PAM recordings contain rain, wind, vegetation rustling, and hardware self-noise.
3. **Overlapping Vocalizations**: Multiple birds calling simultaneously cause acoustic masking and spectral overlap.

---

## 3. Engineering Risks & ML Mitigation Strategies

| PAM Risk | Impact on ML Model | Architectural Mitigation |
| :--- | :--- | :--- |
| **Small Labelled Dev Set** | High risk of overfitting to specific aviary acoustic impulse responses. | Use domain-generalizable acoustic features (MFCCs, spectral shape, BirdNET embeddings) + cross-aviary validation. |
| **Hardware Clipping & Loudness Skew** | Amplitude variations due to bird distance from microphone. | Peak/RMS audio normalization in preprocessing pipeline. |
| **Silent / Empty Chunks** | Zero-inflated count targets causing prediction bias. | Silence trimming & activity detection metrics (RMS thresholding). |
| **Sample Rate Mismatches** | Frequency scaling artifacts in STFT/Mel features. | Automated sample rate validation & resampling in `AudioPreprocessor`. |

---

## 4. Conclusion & Sprint 1 Readiness
The dataset infrastructure, validation pipeline (`AudioValidator`), and feature extraction foundation (`FeatureStore` + Plugin Architecture) ensure that all future modelling sprints operate on validated, standardized, and reproducible feature stores.
