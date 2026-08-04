import numpy as np
from scipy import stats
from typing import Dict, Any


def evaluate_residual_diagnostics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    """Computes statistical diagnostics on model error residuals."""
    residuals = y_true - y_pred

    # Normality Test (Shapiro-Wilk if n < 5000, else Kolmogorov-Smirnov)
    if len(residuals) >= 3:
        if len(residuals) < 5000:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
        else:
            shapiro_stat, shapiro_p = stats.kstest(residuals, "norm")
    else:
        _shapiro_stat, shapiro_p = 0.0, 1.0

    # Heteroscedasticity Correlation Test (Spearman correlation between |residuals| and predicted)
    hetero_r, hetero_p = (
        stats.spearmanr(y_pred, np.abs(residuals)) if len(y_pred) > 2 else (0.0, 1.0)
    )

    skewness = float(stats.skew(residuals)) if len(residuals) > 2 else 0.0
    kurtosis = float(stats.kurtosis(residuals)) if len(residuals) > 2 else 0.0

    return {
        "mean_residual": round(float(np.mean(residuals)), 4),
        "std_residual": round(float(np.std(residuals)), 4),
        "skewness": round(skewness, 4),
        "kurtosis": round(kurtosis, 4),
        "normality_pvalue": round(float(shapiro_p), 5),
        "is_normally_distributed": bool(shapiro_p > 0.05),
        "heteroscedasticity_pvalue": round(float(hetero_p), 5),
        "is_homoscedastic": bool(hetero_p > 0.05),
    }
