import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict
from src.config.schema import AppConfig
from src.utils.io import ensure_dir
from src.utils.logging import setup_logger

logger = setup_logger("ModelVisualization")


class ModelPlotter:
    """Automated Machine Learning Model Visualization Suite."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.figures_dir = ensure_dir(config.paths.figures_dir)
        self.dpi = getattr(config.eda, "dpi", 200)

    def plot_predictions_vs_actual(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str
    ):
        """Generates Ground Truth vs Predicted Scatter Plot."""
        out_path = self.figures_dir / "predictions_vs_actual.png"
        fig, ax = plt.subplots(figsize=(7, 6))

        ax.scatter(
            y_true,
            y_pred,
            color="#1f77b4",
            alpha=0.7,
            edgecolors="k",
            s=60,
            label="Audio Sample Predictions",
        )
        max_val = max(np.max(y_true), np.max(y_pred)) + 1
        ax.plot(
            [0, max_val],
            [0, max_val],
            "r--",
            linewidth=2,
            label="Ideal Perfect Prediction (y = x)",
        )

        ax.set_title(
            f"Ground Truth vs Predicted Bird Count ({model_name})",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Actual Bird Count", fontsize=11)
        ax.set_ylabel("Predicted Bird Count", fontsize=11)
        ax.legend()
        plt.tight_layout()
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        logger.info(f"Saved predictions vs actual plot to '{out_path}'.")

    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """Generates Residual Distribution & Error Histograms."""
        out_path = self.figures_dir / "residual_distributions.png"
        residuals = y_true - y_pred

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(residuals, kde=True, color="#d62728", ax=ax)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
        ax.set_title(
            f"Model Residuals Distribution (Mean Error: {np.mean(residuals):.3f})",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Residual (Ground Truth - Prediction)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)

        plt.tight_layout()
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        logger.info(f"Saved residual distribution plot to '{out_path}'.")

    def plot_feature_importances(
        self, importances: Dict[str, float], model_name: str, top_n: int = 15
    ):
        """Generates Feature Importance Bar Chart."""
        if not importances:
            return

        out_path = self.figures_dir / "feature_importances.png"
        df_imp = pd.DataFrame(
            list(importances.items()), columns=["feature", "importance"]
        )
        df_imp.sort_values(by="importance", ascending=False, inplace=True)
        df_top = df_imp.head(top_n)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x="importance", y="feature", data=df_top, palette="viridis", ax=ax)
        ax.set_title(
            f"Top {top_n} Feature Importances ({model_name})",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Relative Importance Score", fontsize=11)

        plt.tight_layout()
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        logger.info(f"Saved feature importances plot to '{out_path}'.")

    def plot_leaderboard_comparison(self, df_leaderboard: pd.DataFrame):
        """Generates Model Leaderboard MAE Bar Chart Comparison."""
        out_path = self.figures_dir / "model_leaderboard_comparison.png"
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.barplot(
            x="cv_mae_mean", y="model_name", data=df_leaderboard, palette="mako", ax=ax
        )
        ax.set_title(
            "Machine Learning Models Out-of-Fold MAE Comparison",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Mean Absolute Error (Lower is Better)", fontsize=11)
        ax.set_ylabel("Model Name", fontsize=11)

        for p in ax.patches:
            width = p.get_width()
            ax.annotate(
                f"{width:.3f}",
                (width + 0.02, p.get_y() + p.get_height() / 2),
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        logger.info(f"Saved model leaderboard comparison plot to '{out_path}'.")
