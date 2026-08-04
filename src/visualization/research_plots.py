import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from typing import List
from src.config.schema import AppConfig
from src.utils.io import ensure_dir
from src.utils.logging import setup_logger

logger = setup_logger("ResearchPlotter")


class ResearchPlotter:
    """Publication-Quality Research Visualizations Engine (300 DPI)."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.figures_dir = ensure_dir(config.paths.figures_dir)
        self.dpi = 300

    def plot_qq_plot(self, residuals: np.ndarray, model_name: str):
        """Generates Residual Q-Q Plot against theoretical normal distribution."""
        out_path = self.figures_dir / "qq_plot.png"
        fig, ax = plt.subplots(figsize=(6, 6))

        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(
            f"Residual Q-Q Plot ({model_name})", fontsize=12, fontweight="bold"
        )
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        logger.info(f"Saved Q-Q plot to '{out_path}'.")

    def plot_shap_beeswarm(
        self, shap_values: np.ndarray, feature_names: List[str], top_n: int = 12
    ):
        """Generates SHAP summary/beeswarm surrogate plot."""
        out_path = self.figures_dir / "shap_beeswarm.png"

        # Calculate mean absolute SHAP value per feature
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        idx_sorted = np.argsort(mean_abs_shap)[::-1][:top_n]

        top_names = [feature_names[i] for i in idx_sorted]
        top_scores = mean_abs_shap[idx_sorted]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=top_scores, y=top_names, palette="magma", ax=ax)
        ax.set_title(
            "Global SHAP Feature Importance Summary", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel(
            "mean(|SHAP value|) (Average Impact on Model Output)", fontsize=11
        )

        plt.tight_layout()
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        logger.info(f"Saved SHAP beeswarm plot to '{out_path}'.")

    def plot_ablation_comparison(self, df_ablation: pd.DataFrame):
        """Generates Feature Ablation Performance Comparison Bar Plot."""
        out_path = self.figures_dir / "ablation_comparison.png"
        fig, ax = plt.subplots(figsize=(9, 5))

        sns.barplot(
            x="cv_mae", y="feature_subset", data=df_ablation, palette="rocket", ax=ax
        )
        ax.set_title(
            "Feature Category Ablation Comparison (MAE)", fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Out-of-Fold Mean Absolute Error (Lower is Better)", fontsize=11)

        for p in ax.patches:
            width = p.get_width()
            ax.annotate(
                f"{width:.3f}",
                (width + 0.01, p.get_y() + p.get_height() / 2),
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        logger.info(f"Saved ablation comparison plot to '{out_path}'.")

    def plot_robustness_degradation(self, df_robustness: pd.DataFrame):
        """Generates Robustness Degradation Plot under Feature Corruptions."""
        out_path = self.figures_dir / "robustness_degradation.png"
        fig, ax = plt.subplots(figsize=(9, 5))

        sns.lineplot(
            data=df_robustness,
            x="severity_level",
            y="cv_mae",
            hue="perturbation_type",
            marker="o",
            linewidth=2.5,
            ax=ax,
        )
        ax.set_title(
            "Model Error Degradation under Feature Perturbations",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Perturbation Severity Level", fontsize=11)
        ax.set_ylabel("Out-of-Fold MAE", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        logger.info(f"Saved robustness degradation plot to '{out_path}'.")

    def plot_calibration(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """Generates Calibration / Reliability Plot."""
        out_path = self.figures_dir / "calibration_plot.png"
        fig, ax = plt.subplots(figsize=(7, 6))

        ax.scatter(
            y_true,
            y_pred,
            alpha=0.6,
            color="#2ca02c",
            edgecolors="k",
            s=50,
            label="Predictions",
        )
        max_val = max(np.max(y_true), np.max(y_pred)) + 1
        ax.plot([0, max_val], [0, max_val], "k--", label="Perfect Calibration Line")

        ax.set_title(
            f"Model Count Calibration Curve ({model_name})",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Ground Truth Bird Count", fontsize=11)
        ax.set_ylabel("Calibrated Model Prediction", fontsize=11)
        ax.legend()

        plt.tight_layout()
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        logger.info(f"Saved calibration plot to '{out_path}'.")
