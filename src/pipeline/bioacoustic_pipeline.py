import time
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from src.config.schema import AppConfig
from src.config.schemas import (
    AudioFileContract,
    ValidationContract,
    PreprocessingContract,
    FeatureRecordContract,
)
from src.data.validator import AudioValidator
from src.data.preprocessing import AudioPreprocessor
from src.data.versioning import DatasetVersionManager
from src.visualization.eda_generator import EDAGenerator
from src.features.registry import get_extractor, list_registered_extractors
from src.features.feature_store import FeatureStore
from src.features.quality_analyzer import FeatureQualityAnalyzer
from src.features.selection import get_feature_selector
from src.models.artifact_registry import ArtifactRegistry
from src.evaluation.benchmark import PipelineTelemetry
from src.evaluation.feature_benchmark import FeatureBenchmarkSuite
from src.utils.diagram_generator import generate_architecture_svg
from src.utils.io import ensure_dir
from src.utils.logging import setup_logger
from src.visualization.feature_plots import FeaturePlotter


class BioAcousticPipeline:
    """Enterprise Bioacoustic Master Pipeline Orchestrator for Sprint 2 Feature Platform."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.logs_dir = ensure_dir(config.paths.logs_dir)

        # Specialized Loggers for Rich Logging
        self.logger = setup_logger("BioAcousticPipeline", log_file=self.logs_dir / "pipeline.log")
        self.feature_logger = setup_logger("FeatureExtraction", log_file=self.logs_dir / "feature.log")

        # Enterprise Registries, Telemetry & Feature Store v2
        self.telemetry = PipelineTelemetry(config)
        self.artifact_registry = ArtifactRegistry(config)
        self.version_manager = DatasetVersionManager(config)
        self.feature_store_v2 = FeatureStore(config)
        self.quality_analyzer = FeatureQualityAnalyzer(config)
        self.feature_plotter = FeaturePlotter(config)

    def run_validation(self) -> pd.DataFrame:
        """Stage 1: Audio Validation with Contract Enforcement."""
        t0 = time.time()
        self.logger.info("=== STAGE 1: AUDIO VALIDATION & CONTRACT ENFORCEMENT ===")
        validator = AudioValidator(self.config)
        val_df = validator.validate_dataset()

        # Enforce Validation Data Contract
        if not val_df.empty:
            for _, row in val_df.iterrows():
                try:
                    ValidationContract(
                        file_path=str(row["file_path"]),
                        filename=str(row["filename"]),
                        status=str(row["status"]),
                        error_msg=str(row.get("error_msg", "")),
                        sample_rate=int(row.get("sample_rate", 0)),
                        channels=int(row.get("channels", 0)),
                        duration_sec=float(row.get("duration_sec", 0.0)),
                        file_hash=str(row.get("file_hash", "")),
                    )
                except Exception as e:
                    self.logger.error(f"Validation Contract Violation on file '{row.get('filename')}': {e}")
                    if self.config.validation.strict_mode:
                        raise ValueError(f"Strict Mode Contract Violation: {e}")

        # Register artifact
        val_report_path = Path(self.config.validation.report_file)
        if val_report_path.exists():
            self.artifact_registry.register_artifact(
                val_report_path,
                category="validation",
                artifact_name="validation_report",
                version=self.config.project.version,
            )

        self.telemetry.record_stage_runtime("validation", time.time() - t0)
        return val_df

    def run_preprocessing(self) -> int:
        """Stage 2: Audio Preprocessing."""
        t0 = time.time()
        self.logger.info("=== STAGE 2: AUDIO PREPROCESSING ===")
        preprocessor = AudioPreprocessor(self.config)
        count = preprocessor.process_dataset()
        self.telemetry.record_stage_runtime("preprocessing", time.time() - t0)
        return count

    def run_eda(self, val_df: pd.DataFrame) -> Path:
        """Stage 3: Automated EDA & Enterprise Data Quality Dashboard."""
        t0 = time.time()
        self.logger.info("=== STAGE 3: EDA & DATA QUALITY DASHBOARD ===")
        eda = EDAGenerator(self.config)
        eda_report = eda.generate_eda_report()
        dashboard = eda.generate_quality_dashboard(val_df)

        if dashboard.exists():
            self.artifact_registry.register_artifact(
                dashboard,
                category="eda",
                artifact_name="quality_dashboard",
                version=self.config.project.version,
            )

        self.telemetry.record_stage_runtime("eda", time.time() - t0)
        return eda_report

    def run_feature_extraction(self) -> Tuple[pd.DataFrame, List[Any]]:
        """Stage 4: Feature Extraction Plugin Registry Execution."""
        t0 = time.time()
        self.logger.info("=== STAGE 4: FEATURE EXTRACTION PLATFORM V2 ===")
        processed_dir = Path(self.config.paths.processed_data_dir)
        raw_dir = Path(self.config.paths.raw_data_dir)

        target_dir = processed_dir if processed_dir.exists() and any(processed_dir.iterdir()) else raw_dir
        audio_files = []
        for ext in self.config.audio.valid_extensions:
            audio_files.extend(list(target_dir.rglob(f"*{ext}")))

        if not audio_files:
            self.feature_logger.warning(f"No audio files found in '{target_dir}' for feature extraction.")
            return pd.DataFrame(), []

        # Instantiate active extractor plugins
        extractors = []
        for name in self.config.features.active_extractors:
            try:
                ext_inst = get_extractor(
                    name,
                    n_mfcc=self.config.features.n_mfcc,
                    n_fft=self.config.features.n_fft,
                    hop_length=self.config.features.hop_length,
                )
                extractors.append(ext_inst)
            except KeyError as e:
                self.feature_logger.error(f"Extractor load error for '{name}': {e}")

        meta_df = None
        meta_path = raw_dir / self.config.paths.metadata_filename
        if meta_path.exists():
            try:
                meta_df = pd.read_csv(meta_path)
            except Exception as e:
                self.feature_logger.warning(f"Could not read metadata.csv: {e}")

        records = []
        for file_path in audio_files:
            try:
                y, sr = librosa.load(file_path, sr=self.config.audio.target_sample_rate)
            except Exception as e:
                self.feature_logger.error(f"Failed loading '{file_path.name}': {e}")
                continue

            record = {
                "file_path": str(file_path),
                "filename": file_path.name,
                "duration_sec": round(len(y) / sr, 3),
                "sample_rate": sr,
            }

            if meta_df is not None and "filename" in meta_df.columns:
                match = meta_df[meta_df["filename"] == file_path.name]
                if not match.empty:
                    if "bird_count" in match.columns:
                        record["bird_count"] = int(match.iloc[0]["bird_count"])
                    if "species" in match.columns:
                        record["species"] = str(match.iloc[0]["species"])

            # Extract features across all active plugins
            for ext_inst in extractors:
                feats = ext_inst.extract(y, sr)
                record.update(feats)

            # Contract Validation
            try:
                FeatureRecordContract(
                    file_path=record["file_path"],
                    filename=record["filename"],
                    duration_sec=record["duration_sec"],
                    sample_rate=record["sample_rate"],
                    bird_count=record.get("bird_count"),
                    species=record.get("species"),
                    feature_count=len(record) - 4,
                )
            except Exception as e:
                self.feature_logger.error(f"Feature Record Contract Error on '{file_path.name}': {e}")

            records.append(record)

        feature_df = pd.DataFrame(records)
        self.feature_logger.info(
            f"Extracted {feature_df.shape[1]} features across {len(feature_df)} recordings."
        )
        self.telemetry.record_stage_runtime("feature_extraction", time.time() - t0)
        return feature_df, extractors

    def run_feature_quality_and_selection(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Stage 5: Quality Analysis & Configurable Feature Selection."""
        t0 = time.time()
        self.logger.info("=== STAGE 5: FEATURE QUALITY ANALYSIS & SELECTION ===")
        if feature_df.empty:
            return feature_df

        # 1. Quality Analysis Report
        self.quality_analyzer.analyze_features(feature_df)

        # 2. Automated Feature Selection
        selector = get_feature_selector(
            self.config.features.feature_selector,
            threshold=self.config.features.correlation_threshold,
        )
        selected_cols = selector.select_features(feature_df, target_col="bird_count")

        # Keep meta columns + selected features
        meta_cols = ["file_path", "filename", "species", "bird_count"]
        keep_cols = [c for c in meta_cols if c in feature_df.columns] + selected_cols
        filtered_df = feature_df[keep_cols].copy()

        # 3. Visualizations Suite
        self.feature_plotter.generate_all_plots(filtered_df, target_col="bird_count")

        self.telemetry.record_stage_runtime("quality_and_selection", time.time() - t0)
        return filtered_df

    def run_feature_store_v2(self, feature_df: pd.DataFrame, extractors: List[Any]) -> Optional[Dict[str, Path]]:
        """Stage 6: Feature Store v2 Bundle Export."""
        t0 = time.time()
        self.logger.info("=== STAGE 6: FEATURE STORE V2 EXPORT ===")
        if feature_df.empty:
            return None

        ext_meta = {ext.name: ext.get_metadata() for ext in extractors}
        artifacts = self.feature_store_v2.save_feature_bundle(
            feature_df,
            dataset_name="biodcase_avian_features",
            version=self.config.project.version,
            normalization_method=self.config.features.normalization_method,
            extractor_metadata=ext_meta,
        )

        # Register Artifacts
        self.artifact_registry.register_artifact(
            artifacts["norm_parquet"],
            category="feature_store",
            artifact_name="avian_features_norm",
            version=self.config.project.version,
        )

        self.telemetry.record_stage_runtime("feature_store_v2", time.time() - t0)
        return artifacts

    def run_full_pipeline(self) -> bool:
        """Executes complete Sprint 2 feature engineering platform pipeline."""
        self.logger.info(f"Starting Sprint 2 Enterprise Pipeline Execution [{self.config.project.environment.upper()}]...")

        # 0. Generate Architecture SVG
        generate_architecture_svg(Path(self.config.paths.reports_dir) / "architecture.svg")

        # Stage 1: Validation
        val_df = self.run_validation()

        # Stage 2: Preprocessing
        self.run_preprocessing()

        # Stage 3: EDA & Data Quality Dashboard
        self.run_eda(val_df)

        # Stage 4: Feature Extraction
        feats_df, extractors = self.run_feature_extraction()

        # Stage 5: Feature Quality & Selection
        selected_feats_df = self.run_feature_quality_and_selection(feats_df)

        # Stage 6: Feature Store v2 Export
        self.run_feature_store_v2(selected_feats_df, extractors)

        # Stage 7: Extractor Benchmarking
        if extractors:
            suite = FeatureBenchmarkSuite(self.config)
            suite.run_suite_and_export(extractors)

        # Stage 8: Dataset Versioning Manifests
        self.version_manager.create_version_manifest(
            version=self.config.project.version,
            description=f"BioDCASE {self.config.project.environment} feature platform release",
        )

        # Stage 9: Telemetry & Pipeline Metadata
        self.telemetry.generate_benchmark_report(num_files_processed=len(selected_feats_df))
        self.telemetry.generate_pipeline_metadata(
            dataset_version=self.config.project.version,
            num_files=len(selected_feats_df),
            status="SUCCESS",
        )

        self.logger.info("=== SPRINT 2 ENTERPRISE FEATURE PLATFORM EXECUTION COMPLETE ===")
        return True
