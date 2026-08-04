import numpy as np
import pandas as pd
import scipy.stats as stats
from pathlib import Path
from typing import Dict, Any, Optional
from src.config.schema import AppConfig
from src.utils.io import ensure_dir
from src.utils.logging import setup_logger

logger = setup_logger("FeatureQualityAnalyzer")


class FeatureQualityAnalyzer:
    """Enterprise Feature Quality & Distribution Analyzer."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.reports_dir = ensure_dir(config.paths.reports_dir)

    def analyze_features(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = "bird_count",
        correlation_threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """Performs full statistical evaluation of extracted features."""
        # Isolate numerical feature columns
        non_feat_cols = ["file_path", "filename", "species", target_col]
        feature_cols = [
            c
            for c in df.columns
            if c not in non_feat_cols and np.issubdtype(df[c].dtype, np.number)
        ]

        if not feature_cols:
            logger.warning("No numeric feature columns found for quality analysis.")
            return {}

        X = df[feature_cols].copy()
        y = df[target_col] if target_col in df.columns else None

        # 1. Missing Values
        missing = X.isnull().sum().to_dict()
        total_rows = len(X)

        # 2. Variance & Normalization Stats
        variances = X.var().to_dict()
        means = X.mean().to_dict()
        stds = X.std().fillna(0).to_dict()
        mins = X.min().to_dict()
        maxs = X.max().to_dict()

        # 3. Distribution Stats (Skewness & Kurtosis)
        skewness = X.apply(
            lambda col: float(stats.skew(col.dropna()))
            if len(col.dropna()) > 2
            else 0.0
        ).to_dict()
        kurtosis = X.apply(
            lambda col: float(stats.kurtosis(col.dropna()))
            if len(col.dropna()) > 2
            else 0.0
        ).to_dict()

        # 4. Correlation & Redundancy Analysis
        corr_matrix = X.corr().abs()
        collinear_pairs = []
        for i in range(len(feature_cols)):
            for j in range(i + 1, len(feature_cols)):
                f1, f2 = feature_cols[i], feature_cols[j]
                val = corr_matrix.loc[f1, f2]
                if not np.isnan(val) and val >= correlation_threshold:
                    collinear_pairs.append((f1, f2, round(float(val), 3)))

        redundancy_index = round(
            len(collinear_pairs)
            / max(1, len(feature_cols) * (len(feature_cols) - 1) / 2),
            4,
        )

        # 5. Outliers (Z-score > 3.0)
        outlier_counts = {}
        for col in feature_cols:
            col_std = stds[col]
            if col_std > 1e-7:
                z_scores = np.abs((X[col] - means[col]) / col_std)
                outlier_counts[col] = int((z_scores > 3.0).sum())
            else:
                outlier_counts[col] = 0

        # 6. Mutual Information with target (if available)
        mi_scores = {}
        if y is not None and y.notnull().any():
            try:
                from sklearn.feature_selection import mutual_info_regression

                y_clean = y.fillna(y.median())
                X_clean = X.fillna(X.median())
                scores = mutual_info_regression(X_clean, y_clean, random_state=42)
                mi_scores = {
                    col: round(float(score), 4)
                    for col, score in zip(feature_cols, scores)
                }
            except Exception as e:
                logger.warning(f"Could not compute mutual information: {e}")

        # 7. Low Variance Features (near zero variance)
        low_variance_features = [col for col, var in variances.items() if var < 1e-6]

        summary = {
            "total_features": len(feature_cols),
            "total_samples": total_rows,
            "missing_values": missing,
            "variances": variances,
            "means": means,
            "stds": stds,
            "mins": mins,
            "maxs": maxs,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "collinear_pairs_count": len(collinear_pairs),
            "collinear_pairs": collinear_pairs,
            "redundancy_index": redundancy_index,
            "outlier_counts": outlier_counts,
            "mutual_information": mi_scores,
            "low_variance_features": low_variance_features,
        }

        self.generate_report_markdown(summary)
        return summary

    def generate_report_markdown(self, summary: Dict[str, Any]) -> Path:
        """Renders comprehensive markdown report: reports/feature_quality_report.md."""
        report_path = self.reports_dir / "feature_quality_report.md"

        low_var_str = (
            ", ".join(summary["low_variance_features"])
            if summary["low_variance_features"]
            else "None"
        )
        top_collinear = summary["collinear_pairs"][:10]

        collinear_table = ""
        for f1, f2, r in top_collinear:
            collinear_table += f"| `{f1}` | `{f2}` | **{r}** |\n"
        if not collinear_table:
            collinear_table = "| None | None | N/A |\n"

        content = f"""# Enterprise Feature Quality Analysis Report

**Project**: {self.config.project.name}  
**Environment**: {self.config.project.environment}  
**Evaluated Features**: {summary['total_features']}  
**Evaluated Samples**: {summary['total_samples']}  
**Redundancy Index**: **{summary['redundancy_index']}**  

---

## 1. Quality Overview & Health Checks

| Check | Result | Threshold / Standard | Status |
| :--- | :--- | :--- | :--- |
| **Missing Values** | {sum(summary['missing_values'].values())} total | 0 | {"PASS" if sum(summary['missing_values'].values()) == 0 else "FAIL"} |
| **Low-Variance Features** | {len(summary['low_variance_features'])} features | 0 | {"PASS" if len(summary['low_variance_features']) == 0 else "WARNING"} |
| **Highly Collinear Pairs** | {summary['collinear_pairs_count']} pairs | < 10 | {"PASS" if summary['collinear_pairs_count'] < 10 else "WARNING"} |
| **Feature Redundancy Ratio** | {summary['redundancy_index']} | < 0.20 | {"PASS" if summary['redundancy_index'] < 0.20 else "WARNING"} |

---

## 2. Low-Variance & Uninformative Features
- **Flagged Features (Var < 1e-6)**: `{low_var_str}`

---

## 3. High Collinearity & Feature Redundancy (> 0.85 Pearson Correlation)

| Feature 1 | Feature 2 | Absolute Correlation |
| :--- | :--- | :--- |
{collinear_table}

---

## 4. Top Features by Mutual Information (Target Correlation)
"""
        if summary.get("mutual_information"):
            sorted_mi = sorted(
                summary["mutual_information"].items(), key=lambda x: x[1], reverse=True
            )[:10]
            content += (
                "\n| Feature Name | Mutual Information Score |\n| :--- | :--- |\n"
            )
            for f_name, score in sorted_mi:
                content += f"| `{f_name}` | **{score}** |\n"
        else:
            content += "\n*Mutual Information calculation skipped (no target labels or single class).*\n"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Feature quality analysis report saved to '{report_path}'.")
        return report_path
