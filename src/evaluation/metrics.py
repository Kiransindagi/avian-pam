import numpy as np
from scipy import stats
from typing import Dict, Any


def compute_avian_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Calculates research-grade statistical metrics with 95% bootstrap confidence intervals."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    medae = float(np.median(np.abs(residuals)))

    # MAPE (safely handling zero targets)
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = float(
            np.mean(np.abs(residuals[non_zero_mask] / y_true[non_zero_mask])) * 100.0
        )
    else:
        mape = 0.0

    # R^2 Score
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / max(1e-7, ss_tot)))

    # Pearson & Spearman Correlation Coefficients
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
        spearman_rho, spearman_p = stats.spearmanr(y_true, y_pred)
    else:
        pearson_r, pearson_p = 0.0, 1.0
        spearman_rho, spearman_p = 0.0, 1.0

    prediction_bias = float(np.mean(y_pred - y_true))
    prediction_variance = float(np.var(y_pred))

    # Bootstrap 95% Confidence Intervals for MAE and RMSE
    rng = np.random.RandomState(random_state)
    boot_maes, boot_rmses = [], []
    n_samples = len(y_true)

    if n_samples >= 5:
        for _ in range(n_bootstrap):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            b_true, b_pred = y_true[indices], y_pred[indices]
            boot_maes.append(np.mean(np.abs(b_true - b_pred)))
            boot_rmses.append(np.sqrt(np.mean((b_true - b_pred) ** 2)))

        mae_ci_lower = float(np.percentile(boot_maes, 2.5))
        mae_ci_upper = float(np.percentile(boot_maes, 97.5))
        rmse_ci_lower = float(np.percentile(boot_rmses, 2.5))
        rmse_ci_upper = float(np.percentile(boot_rmses, 97.5))
    else:
        mae_ci_lower, mae_ci_upper = mae, mae
        rmse_ci_lower, rmse_ci_upper = rmse, rmse

    return {
        "mae": round(mae, 4),
        "mae_ci_95": [round(mae_ci_lower, 4), round(mae_ci_upper, 4)],
        "rmse": round(rmse, 4),
        "rmse_ci_95": [round(rmse_ci_lower, 4), round(rmse_ci_upper, 4)],
        "medae": round(medae, 4),
        "mape": round(mape, 2),
        "r2": round(r2, 4),
        "pearson_r": round(float(pearson_r), 4),
        "pearson_p": round(float(pearson_p), 5),
        "spearman_rho": round(float(spearman_rho), 4),
        "spearman_p": round(float(spearman_p), 5),
        "prediction_bias": round(prediction_bias, 4),
        "prediction_variance": round(prediction_variance, 4),
    }
