import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from src.config.schema import AppConfig
from src.models.model_registry import get_model
from src.models.base_model import BaseAvianModel
from src.training.cross_validation import CrossValidationEngine
from src.models.model_registry import ModelRegistryManager
from src.training.tracker import ExperimentTracker
from src.evaluation.model_benchmark import ModelBenchmarker
from src.visualization.model_plots import ModelPlotter
from src.utils.logging import setup_logger

logger = setup_logger("ModelTrainer")


class ModelTrainer:
    """Enterprise Master Model Trainer & Experiment Orchestrator for Sprint 3."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.registry_manager = ModelRegistryManager(config)
        self.experiment_tracker = ExperimentTracker(config)
        self.benchmarker = ModelBenchmarker(config)
        self.plotter = ModelPlotter(config)
        self.cv_engine = CrossValidationEngine(strategy="group_kfold", n_splits=3)

    def load_latest_features(
        self,
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Loads normalized features and target counts from Feature Store v2."""
        feature_store_dir = Path(self.config.paths.feature_store_dir)
        parquet_files = sorted(list(feature_store_dir.glob("*_norm_*.parquet")))

        if not parquet_files:
            logger.warning(
                f"No normalized parquet features found in '{feature_store_dir}'. Using dummy feature generator."
            )
            return self._generate_dummy_features()

        latest_parquet = parquet_files[-1]
        logger.info(f"Loading feature dataset from '{latest_parquet.name}'...")
        df = pd.read_parquet(latest_parquet)

        meta_cols = ["file_path", "filename", "species", "bird_count"]
        feature_cols = [
            c
            for c in df.columns
            if c not in meta_cols and np.issubdtype(df[c].dtype, np.number)
        ]

        X = df[feature_cols].fillna(0)
        y = (
            df["bird_count"]
            if "bird_count" in df.columns
            else pd.Series(np.random.randint(1, 10, size=len(df)))
        )
        groups = df["species"] if "species" in df.columns else None

        return X, y, groups

    def _generate_dummy_features(
        self,
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        rng = np.random.RandomState(42)
        n_samples = 20
        X = pd.DataFrame({f"feature_{i}": rng.randn(n_samples) for i in range(10)})
        y = pd.Series(rng.randint(1, 15, size=n_samples), name="bird_count")
        groups = pd.Series(
            [f"species_{i % 3}" for i in range(n_samples)], name="species"
        )
        return X, y, groups

    def train_and_evaluate_all_models(self) -> pd.DataFrame:
        """Trains, optimizes, benchmarks, registers, and visualizes all registered ML models."""
        X, y, groups = self.load_latest_features()

        model_names = [
            "dummy_mean",
            "dummy_median",
            "linear_regression",
            "ridge",
            "lasso",
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "xgboost",
            "lightgbm",
            "catboost",
            "svr",
            "knn",
            "voting_ensemble",
            "stacking_ensemble",
        ]

        instantiated_models: List[BaseAvianModel] = []
        for name in model_names:
            try:
                model_inst = get_model(name)
                # Fit model on entire dataset for production checkpointing
                model_inst.fit(X, y)
                instantiated_models.append(model_inst)

                # Register checkpoint
                cv_res = self.cv_engine.evaluate_model(model_inst, X, y, groups=groups)
                self.registry_manager.register_model_checkpoint(
                    model_inst,
                    metrics=cv_res,
                    feature_set_name="biodcase_avian_features",
                    dataset_version=self.config.project.version,
                )

                # Log Experiment Run
                self.experiment_tracker.log_experiment_run(
                    experiment_id=f"exp_sprint3_{name}",
                    model_name=name,
                    hyperparameters=model_inst.get_parameters().get(
                        "hyperparameters", {}
                    ),
                    metrics=cv_res,
                    dataset_version=self.config.project.version,
                )

            except Exception as e:
                logger.error(f"Failed model training for '{name}': {e}")

        # Benchmark Models & Render Leaderboard
        df_leaderboard, _ = self.benchmarker.benchmark_models(
            instantiated_models, X, y, groups=groups
        )

        # Generate Visualizations
        if instantiated_models:
            top_model = instantiated_models[0]
            preds = top_model.predict(X)
            self.plotter.plot_predictions_vs_actual(y.to_numpy(), preds, top_model.name)
            self.plotter.plot_residuals(y.to_numpy(), preds, top_model.name)
            self.plotter.plot_leaderboard_comparison(df_leaderboard)

            imp = top_model.get_feature_importance()
            if imp:
                self.plotter.plot_feature_importances(imp, top_model.name)

        logger.info("=== SPRINT 3 MASTER MODEL TRAINING & BENCHMARKING COMPLETE ===")
        return df_leaderboard
