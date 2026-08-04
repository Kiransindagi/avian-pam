# Changelog

All notable changes to the Avian Passive Acoustic Monitoring (PAM) Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-04

### Added
- Multi-domain feature engineering platform incorporating DSP spectral metrics, ecoacoustic soundscape indices, and deep bioacoustic embeddings (BirdNET & PANNs).
- Leakage-free Group K-Fold cross-validation framework grouped by aviary environment.
- Benchmarks for 15 ML model architectures including tree ensembles, linear models, SVR, KNN, and stacking meta-learners.
- Comprehensive scientific evaluation suite featuring statistical hypothesis testing, SHAP explainability, feature category ablation, and perturbation stress-testing.
- Production FastAPI REST API and Docker containerization.
- Unified command-line interface (`cli.py`).
