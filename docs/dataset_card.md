# BioDCASE 2026 Bird Counting Dataset Card

## 1. Dataset Overview
- **Dataset Name**: `Emreargin/BioDCASE2026_Bird_Counting`
- **Repository**: HuggingFace Datasets
- **Domain**: Bioacoustics / Passive Acoustic Monitoring (PAM)
- **Primary Objective**: Estimate bird population counts from multichannel / single-channel field acoustic recordings.

---

## 2. Dataset Structure & Features
- **Raw Format**: 16-bit PCM WAV audio files ($22,050 \text{ Hz}$ or $44,100 \text{ Hz}$ sampling rates).
- **Labels**: Discrete integer ground-truth bird counts.
- **Metadata**: Species labels, aviary IDs, recording duration (sec), sample rate.

---

## 3. Data Processing & Health
- **Validation Pipeline**: Verified format contracts, sample rate conversions, and corruption-free audio streams using `DataValidator`.
- **Quality Analysis**: Zero corrupted files, verified duration distributions, and balanced species coverage.
- **Version Control**: Versioned under `data/versions/manifest_v2.0.0.json` with SHA-256 integrity checksums.
