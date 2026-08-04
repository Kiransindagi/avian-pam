# Avian Population Estimation Model Card

## 1. Model Details
- **Developer**: Avian Bioacoustics ML Engineering Team
- **Model Architecture**: Linear / Tree-Ensemble Regressors (RandomForest, ExtraTrees, GradientBoosting, XGBoost, LightGBM, CatBoost) & Stacking Regressors.
- **Model Version**: `v2.0.0`
- **License**: MIT
- **Task**: Passive Acoustic Monitoring (PAM) Bird Count Regression

---

## 2. Intended Use
- **Primary Intended Use**: Predicting continuous and discrete bird counts from field audio recordings collected in aviary habitats.
- **Out-of-Scope Use Cases**: Non-avian wildlife bioacoustics, speech recognition, underwater acoustic monitoring without retrained domain adaptors.

---

## 3. Training & Evaluation Data
- **Dataset**: `Emreargin/BioDCASE2026_Bird_Counting` (HuggingFace).
- **Features**: DSP representations (Spectrogram, MFCC, Zero-crossing, Spectral Centroid), Ecoacoustic Soundscape Indices (ACI, BI, NDSI, ADI, AEI), Pretrained Audio Embeddings (BirdNET / PANNs).
- **Evaluation Strategy**: Out-of-fold Group K-Fold Cross-Validation preventing data leakage between recordings of the same aviary.

---

## 4. Performance & Metrics
- **Out-of-Fold Mean Absolute Error (MAE)**: **1.383 ± 0.294**
- **Out-of-Fold Root Mean Square Error (RMSE)**: **1.586 ± 0.170**
- **Coefficient of Determination ($R^2$)**: **0.449**
- **Inference Latency**: **< 25 ms per audio file**

---

## 5. Limitations & Ethical Considerations
- **High-Density Chorus Overlap**: Underprediction risk during extreme chorus density (15+ vocalizing individuals simultaneously).
- **Background Environmental Noise**: Wind and ambient rainfall degrade spectral clarity and alter Ecoacoustic soundscape metrics.
