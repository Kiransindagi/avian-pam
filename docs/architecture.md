# System Architecture & Platform Topology

## Overview
The **Avian Passive Acoustic Monitoring (PAM) Platform** is an enterprise-grade, modular machine learning pipeline built to estimate bird population abundance from field acoustic recordings.

---

## 1. High-Level Architecture Topology

```
                  ┌──────────────────────────────────────────────────┐
                  │   BioDCASE Passive Acoustic Field Recordings    │
                  └─────────────────────────┬────────────────────────┘
                                            │
                                            ▼
                  ┌──────────────────────────────────────────────────┐
                  │    Audio Preprocessing & Validation Engine       │
                  └─────────────────────────┬────────────────────────┘
                                            │
                                            ▼
                  ┌──────────────────────────────────────────────────┐
                  │    Feature Engineering Platform (Store v2)       │
                  │  (DSP Features + Ecoacoustics + Deep Embeddings) │
                  └─────────────────────────┬────────────────────────┘
                                            │
                                            ▼
                  ┌──────────────────────────────────────────────────┐
                  │   Model Registry & Group K-Fold CV Engine        │
                  │   (15 Baseline, Tree, Kernel & Stacking Models)  │
                  └─────────────────────────┬────────────────────────┘
                                            │
                                            ▼
                  ┌──────────────────────────────────────────────────┐
                  │   Research Evaluation & Explainability Suite     │
                  │   (Statistical Tests, SHAP, Ablation, Robustness)│
                  └─────────────────────────┬────────────────────────┘
                                            │
                                            ▼
                  ┌──────────────────────────────────────────────────┐
                  │   Production FastAPI REST API & Submission CLI   │
                  └─────────────────────────┴────────────────────────┘
```

---

## 2. Directory & Module Responsibilities

| Package | Primary Responsibility | Key Interfaces |
| :--- | :--- | :--- |
| `src.data` | Data validation, PCM audio normalization, versioning manifests | `AudioValidator`, `AudioPreprocessor`, `DatasetVersionManager` |
| `src.features` | Plugin extractor registry, quality filtering, Feature Store v2 | `@register_extractor`, `FeatureStoreV2`, `FeatureQualityAnalyzer` |
| `src.models` | Model interface contracts, algorithm implementations, registry | `BaseAvianModel`, `@register_model`, `ModelRegistryManager` |
| `src.training` | Leakage-free Group K-Fold CV, Random Search HPO, model training | `GroupKFoldTrainer`, `RandomSearchOptimizer`, `ModelTrainer` |
| `src.evaluation` | Hypothesis testing, SHAP explainability, error diagnostics, ablation | `StatisticalSignificanceTester`, `ExplainabilityEngine`, `ErrorAnalyzer` |
| `src.inference` | Production inference engine, FastAPI service, BioDCASE submissions | `AvianInferenceEngine`, `FastAPI_Service`, `BioDCASESubmissionGenerator` |

---

## 3. Data Contracts & Schema Validation
All data streams moving between pipeline stages undergo contract validation (`src/contracts/schemas.py`) using Pydantic models:
- **`AudioFileContract`**: Enforces `.wav`, `.flac`, `.mp3` extension, sample rate ($22,050 \text{ Hz}$), duration, and channel count.
- **`PreprocessingContract`**: Ensures standardized signal output.
- **`FeatureRecordContract`**: Validates non-empty feature vectors and expected column types.

---

## 4. Configuration Management
System configurations are centralized in `configs/`:
- `configs/development.yaml`: Fast iteration and testing parameters.
- `configs/production.yaml`: High-precision production inference parameters.
- `configs/experiment.yaml`: Deep grid search and HPO parameters.
