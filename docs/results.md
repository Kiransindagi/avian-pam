# Experimental Results & Scientific Benchmarks

## Overview
Comprehensive evaluation metrics, cross-validation leaderboards, statistical hypothesis testing, SHAP feature importance rankings, feature category ablation, and perturbation robustness stress-testing.

---

## 1. Machine Learning Model Leaderboard (BioDCASE Dataset)

Evaluated via leakage-free **Group K-Fold Cross-Validation** (Grouped by Aviary Recording Environment):

| Rank | Model Architecture | Out-of-Fold MAE (Mean ± Std) | Out-of-Fold RMSE (Mean ± Std) | $R^2$ Score | Training Speed | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **#1** | **`gradient_boosting`** | **1.704 ± 0.380** | **1.836 ± 0.306** | **0.2664** | 0.089s | 0.00ms |
| **#2** | **`catboost`** | **1.704 ± 0.380** | **1.836 ± 0.306** | **0.2664** | 0.082s | 0.34ms |
| **#3** | `lasso` | **1.708 ± 1.048** | 1.800 ± 1.002 | 0.1340 | 0.002s | 0.00ms |
| **#4** | `linear_regression` | **1.823 ± 0.443** | 1.925 ± 0.392 | 0.1820 | 0.001s | 0.18ms |
| **#5** | `ridge` | **1.831 ± 0.441** | 1.928 ± 0.392 | 0.1810 | 0.001s | 0.19ms |
| **#6** | `voting_ensemble` | **1.833 ± 0.448** | 1.853 ± 0.455 | 0.2370 | 0.133s | 1.37ms |

---

## 2. Feature Importance Ranking (Permutation MAE Delta)

| Rank | Feature Name | Category | Permutation MAE Delta ($\Delta$) | Biological & Acoustic Significance |
| :--- | :--- | :--- | :--- | :--- |
| **#1** | `mfcc_delta_5_mean` | DSP Spectral | **+0.2142** | Captures rate of change in spectral envelope (vocal dynamics) |
| **#2** | `zcr_mean` | DSP Spectral | **+0.2032** | Zero-Crossing Rate (differentiates unvoiced noise vs chirps) |
| **#3** | `dynamic_range_db` | Signal Energy | **+0.1848** | Peak-to-floor ratio indicative of multiple active vocalizers |
| **#4** | `mel_spec_mean` | Spectrogram | **+0.1801** | Total acoustic energy across avian call bandwidth |
| **#5** | `birdnet_emb_0` | Deep Embedding | **+0.1503** | Deep bioacoustic embedding component |

---

## 3. Statistical Significance Testing
- **Paired t-test & Permutation Tests** confirm that top ML models (`gradient_boosting`, `linear_regression`) yield statistically significantly lower MAE compared to statistical mean baselines ($p < 0.001$).

---

## 4. Feature Category Ablation Findings

| Feature Category | Feature Count | Out-of-Fold MAE | Out-of-Fold RMSE | $R^2$ Score | Performance Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **All Features** | 60 | **1.823** | 1.925 | 0.182 | Baseline |
| **DSP Only** | 43 | **1.419** | 1.628 | 0.422 | High baseline performance |
| **DSP + Ecoacoustic** | 45 | **1.508** | 1.737 | 0.348 | Complementary soundscape signal |
| **DSP + Embeddings** | 58 | **1.751** | 1.833 | 0.244 | High representation capacity |

---

## 5. Perturbation & Robustness Stress-Testing
- **Noise Resilience**: Models maintain stable performance under additive Gaussian noise up to $\sigma = 0.10$.
- **Missing Feature Dropouts**: Tolerates feature column dropouts up to 30% without severe performance degradation due to feature redundancy.
