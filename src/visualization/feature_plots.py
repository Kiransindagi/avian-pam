import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from src.config.schema import AppConfig
from src.utils.io import ensure_dir
from src.utils.logging import setup_logger

logger = setup_logger("FeatureVisualization")


class FeaturePlotter:
    """Automated Production Feature Visualization Suite."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.figures_dir = ensure_dir(config.paths.figures_dir)
        self.dpi = getattr(config.eda, "dpi", 200)

    def generate_all_plots(
        self, df: pd.DataFrame, target_col: Optional[str] = "bird_count"
    ):
        """Generates all automated visualization figures."""
        non_feat_cols = ["file_path", "filename", "species", target_col]
        feature_cols = [
            c
            for c in df.columns
            if c not in non_feat_cols and np.issubdtype(df[c].dtype, np.number)
        ]

        if not feature_cols or len(df) == 0:
            logger.warning("DataFrame empty or no numeric feature columns to plot.")
            return

        self.plot_correlation_heatmap(df, feature_cols[:15])
        self.plot_feature_distributions(df, feature_cols[:6])
        self.plot_pca_projection(df, feature_cols, target_col)
        self.plot_tsne_projection(df, feature_cols, target_col)
        self.plot_species_breakdown(df, feature_cols[:4], target_col)

    def plot_correlation_heatmap(self, df: pd.DataFrame, feature_cols: List[str]):
        """Generates correlation matrix heatmap plot."""
        out_path = self.figures_dir / "correlation_heatmap.png"
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df[feature_cols].corr()
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax
        )
        ax.set_title(
            "Acoustic Feature Correlation Matrix", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        logger.info(f"Saved correlation heatmap to '{out_path}'.")

    def plot_feature_distributions(self, df: pd.DataFrame, feature_cols: List[str]):
        """Generates histogram & KDE density plots."""
        out_path = self.figures_dir / "feature_distributions.png"
        n_plots = len(feature_cols)
        cols = 3
        rows = int(np.ceil(n_plots / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows))
        axes = axes.flatten() if n_plots > 1 else [axes]

        for idx, col in enumerate(feature_cols):
            sns.histplot(df[col], kde=True, ax=axes[idx], color="#1f77b4")
            axes[idx].set_title(f"Distribution: {col}", fontsize=11, fontweight="bold")

        for idx in range(n_plots, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        logger.info(f"Saved feature distributions plot to '{out_path}'.")

    def plot_pca_projection(
        self, df: pd.DataFrame, feature_cols: List[str], target_col: Optional[str]
    ):
        """Generates 2D PCA projection scatter plot."""
        out_path = self.figures_dir / "pca_projection.png"
        X = df[feature_cols].fillna(0)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(8, 6))
        color_data = df[target_col] if target_col and target_col in df.columns else None
        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=color_data,
            cmap="viridis",
            alpha=0.8,
            edgecolors="k",
        )
        if color_data is not None:
            plt.colorbar(scatter, label="Bird Population Count")

        ax.set_title(
            f"2D PCA Projection (Exp Var: {np.sum(pca.explained_variance_ratio_):.2f})",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        plt.tight_layout()
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        logger.info(f"Saved PCA projection to '{out_path}'.")

    def plot_tsne_projection(
        self, df: pd.DataFrame, feature_cols: List[str], target_col: Optional[str]
    ):
        """Generates 2D t-SNE embedding projection scatter plot."""
        out_path = self.figures_dir / "umap_projection.png"
        X = df[feature_cols].fillna(0)
        if len(X) < 4:
            return  # Need sufficient samples for t-SNE

        perplexity = min(30, max(2, len(X) - 1))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X)

        fig, ax = plt.subplots(figsize=(8, 6))
        color_data = df[target_col] if target_col and target_col in df.columns else None
        scatter = ax.scatter(
            X_tsne[:, 0],
            X_tsne[:, 1],
            c=color_data,
            cmap="magma",
            alpha=0.8,
            edgecolors="k",
        )
        if color_data is not None:
            plt.colorbar(scatter, label="Bird Population Count")

        ax.set_title("2D t-SNE / Manifold Projection", fontsize=12, fontweight="bold")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        plt.tight_layout()
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        logger.info(f"Saved t-SNE projection to '{out_path}'.")

    def plot_species_breakdown(
        self, df: pd.DataFrame, feature_cols: List[str], target_col: Optional[str]
    ):
        """Generates species / target breakdown boxplots."""
        out_path = self.figures_dir / "species_feature_breakdown.png"
        group_col = "species" if "species" in df.columns else target_col
        if not group_col or group_col not in df.columns:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for idx, col in enumerate(feature_cols[:4]):
            sns.boxplot(x=group_col, y=col, data=df, ax=axes[idx], palette="Blues")
            axes[idx].set_title(f"{col} by {group_col.capitalize()}", fontsize=11)

        plt.tight_layout()
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        logger.info(f"Saved species feature breakdown to '{out_path}'.")
