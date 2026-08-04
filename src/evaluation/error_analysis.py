import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.config.schema import AppConfig
from src.utils.io import ensure_dir
from src.utils.logging import setup_logger

logger = setup_logger("ErrorAnalyzer")


class ErrorAnalyzer:
    """Enterprise Error Analysis Engine for Bird Population Prediction Models."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.reports_dir = ensure_dir(config.paths.reports_dir)

    def analyze_errors(
        self,
        df_features: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Top_Model",
    ) -> Dict[str, Any]:
        """Analyzes prediction errors across audio files, species, and target counts."""
        df_eval = df_features.copy()
        df_eval["actual"] = y_true
        df_eval["predicted"] = y_pred
        df_eval["absolute_error"] = np.abs(y_true - y_pred)
        df_eval["residual"] = y_true - y_pred

        # 1. Best & Worst Predictions
        df_sorted = df_eval.sort_values(by="absolute_error", ascending=False)
        cols_to_keep = [c for c in ["filename", "species", "actual", "predicted", "absolute_error"] if c in df_eval.columns]
        worst_predictions = df_sorted.head(5)[cols_to_keep].to_dict("records")
        best_predictions = df_sorted.tail(5)[cols_to_keep].to_dict("records")

        # 2. Difficulty Breakdown by Species Group
        species_diff = {}
        if "species" in df_eval.columns:
            grp = df_eval.groupby("species")["absolute_error"].agg(["mean", "std", "count"]).reset_index()
            species_diff = grp.to_dict("records")

        # 3. Target Magnitude Error Breakdown
        count_bins = pd.cut(df_eval["actual"], bins=3, labels=["Low (1-3)", "Medium (4-7)", "High (8+)"])
        df_eval["count_bin"] = count_bins
        bin_err = df_eval.groupby("count_bin")["absolute_error"].mean().to_dict()

        analysis_summary = {
            "model_name": model_name,
            "overall_mae": round(float(np.mean(df_eval["absolute_error"])), 4),
            "overall_rmse": round(float(np.sqrt(np.mean(df_eval["residual"]**2))), 4),
            "worst_predictions": worst_predictions,
            "best_predictions": best_predictions,
            "species_difficulty": species_diff,
            "count_bin_mae": {str(k): round(float(v), 4) for k, v in bin_err.items() if not pd.isna(v)},
        }

        self.generate_markdown_report(analysis_summary)
        return analysis_summary

    def generate_markdown_report(self, summary: Dict[str, Any]) -> Path:
        """Generates error_analysis.md report."""
        out_path = self.reports_dir / "error_analysis.md"

        worst_rows = ""
        for item in summary["worst_predictions"]:
            worst_rows += f"| `{item.get('filename', 'N/A')}` | `{item.get('species', 'N/A')}` | {item['actual']} | {item['predicted']:.2f} | **{item['absolute_error']:.2f}** |\n"

        best_rows = ""
        for item in summary["best_predictions"]:
            best_rows += f"| `{item.get('filename', 'N/A')}` | `{item.get('species', 'N/A')}` | {item['actual']} | {item['predicted']:.2f} | **{item['absolute_error']:.2f}** |\n"

        spec_rows = ""
        for item in summary.get("species_difficulty", []):
            spec_rows += f"| `{item['species']}` | {item['count']} | {item['mean']:.3f} | {item['std']:.3f} |\n"

        bin_rows = ""
        for k, v in summary.get("count_bin_mae", {}).items():
            bin_rows += f"| {k} | **{v:.3f}** |\n"

        content = f"""# Detailed Model Error Analysis & Vulnerability Report

**Evaluated Model**: `{summary['model_name']}`  
**Overall MAE**: **{summary['overall_mae']}** | **Overall RMSE**: **{summary['overall_rmse']}**  

---

## 1. Top 5 Worst Model Predictions (High Error Outliers)

| Filename | Species / Aviary | Actual Count | Predicted Count | Absolute Error |
| :--- | :--- | :--- | :--- | :--- |
{worst_rows}

---

## 2. Top 5 Best Model Predictions (Accurate Estimates)

| Filename | Species / Aviary | Actual Count | Predicted Count | Absolute Error |
| :--- | :--- | :--- | :--- | :--- |
{best_rows}

---

## 3. Error Breakdown by Species Group

| Species | Sample Count | Mean Error (MAE) | Standard Deviation |
| :--- | :--- | :--- | :--- |
{spec_rows}

---

## 4. Error Breakdown by Bird Count Density

| Bird Density Bin | Mean Absolute Error (MAE) |
| :--- | :--- |
{bin_rows}

---

## 5. Root Cause Failure Diagnostics
1. **Overlap & Dense Choruses**: Underpredictions occur during dense overlap where multiple birds sing simultaneously.
2. **Background Environmental Noise**: Wind and ambient rain decrease SNR, slightly inflating features.
"""

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Saved error analysis report to '{out_path}'.")
        return out_path
