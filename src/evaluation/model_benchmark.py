import os
import psutil
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
from src.config.schema import AppConfig
from src.models.base_model import BaseAvianModel
from src.training.cross_validation import CrossValidationEngine
from src.utils.io import ensure_dir
from src.utils.logging import setup_logger

logger = setup_logger("ModelBenchmarker")


class ModelBenchmarker:
    """Enterprise Model Benchmarking & Leaderboard Generator."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.reports_dir = ensure_dir(config.paths.reports_dir)
        self.cv_engine = CrossValidationEngine(strategy="group_kfold", n_splits=3)
        self.process = psutil.Process(os.getpid())

    def benchmark_models(
        self,
        models: List[BaseAvianModel],
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None,
    ) -> Tuple[pd.DataFrame, Path]:
        """Evaluates and benchmarks multiple ML models, generating CSV leaderboard and markdown report."""
        leaderboard_rows = []

        logger.info(
            f"Benchmarking {len(models)} models across cross-validation splits..."
        )

        for model in models:
            mem_before = self.process.memory_info().rss / (1024 * 1024)
            cv_res = self.cv_engine.evaluate_model(model, X, y, groups=groups)
            mem_after = self.process.memory_info().rss / (1024 * 1024)

            leaderboard_rows.append(
                {
                    "model_name": model.name,
                    "version": model.version,
                    "cv_mae_mean": cv_res["mean_mae"],
                    "cv_mae_std": cv_res["std_mae"],
                    "cv_rmse_mean": cv_res["mean_rmse"],
                    "cv_rmse_std": cv_res["std_rmse"],
                    "r2_score": cv_res["overall_r2"],
                    "training_time_sec": round(model.training_time_sec, 4),
                    "inference_latency_ms": round(model.inference_latency_ms, 3),
                    "memory_rss_mb": round(max(0.0, mem_after - mem_before), 2),
                }
            )

        df_leaderboard = pd.DataFrame(leaderboard_rows)
        df_leaderboard.sort_values(by="cv_mae_mean", ascending=True, inplace=True)
        df_leaderboard.reset_index(drop=True, inplace=True)

        # 1. Export CSV Leaderboard
        csv_path = self.reports_dir / "model_leaderboard.csv"
        df_leaderboard.to_csv(csv_path, index=False)
        logger.info(f"Model Leaderboard CSV exported to '{csv_path}'.")

        # 2. Render Markdown Report
        md_path = self.generate_markdown_report(df_leaderboard)

        return df_leaderboard, md_path

    def generate_markdown_report(self, df_leaderboard: pd.DataFrame) -> Path:
        """Renders comprehensive model benchmark markdown report."""
        md_path = self.reports_dir / "model_benchmark_report.md"

        best_model = df_leaderboard.iloc[0]["model_name"]
        best_mae = df_leaderboard.iloc[0]["cv_mae_mean"]
        best_r2 = df_leaderboard.iloc[0]["r2_score"]

        table_rows = ""
        for idx, row in df_leaderboard.iterrows():
            rank = idx + 1
            table_rows += (
                f"| **#{rank}** | `{row['model_name']}` | **{row['cv_mae_mean']:.3f} ± {row['cv_mae_std']:.3f}** | "
                f"{row['cv_rmse_mean']:.3f} ± {row['cv_rmse_std']:.3f} | **{row['r2_score']:.3f}** | "
                f"{row['training_time_sec']:.3f}s | {row['inference_latency_ms']:.2f}ms |\n"
            )

        content = f"""# Enterprise Machine Learning Model Leaderboard & Benchmark Report

**Project**: {self.config.project.name}  
**Environment**: {self.config.project.environment}  
**Top Performing Model**: **`{best_model}`**  
**Best CV MAE**: **{best_mae:.4f}**  
**Best $R^2$**: **{best_r2:.4f}**  

---

## 1. Machine Learning Model Leaderboard (Ranked by CV MAE)

| Rank | Model Name | Out-of-Fold MAE | Out-of-Fold RMSE | $R^2$ Score | Train Time | Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{table_rows}

---

## 2. Evaluation Insights & Modeling Strategy
- **Evaluation Protocol**: Leakage-free Group K-Fold Cross-Validation preventing data leakage between recordings of the same aviary environment.
- **Model Diversity**: Evaluated baseline statistical predictors, linear regressors, kernel methods (SVR, KNN), tree ensembles (RandomForest, ExtraTrees, GradientBoosting, XGBoost, LightGBM, CatBoost), and stacking ensembles.
- **Next Steps (Sprint 4)**: Top performing models will undergo statistical significance testing, SHAP explainability analysis, and error diagnosis.
"""

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Model benchmark markdown report exported to '{md_path}'.")
        return md_path
