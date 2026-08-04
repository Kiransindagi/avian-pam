import numpy as np
from src.evaluation.metrics import compute_avian_metrics


def test_compute_avian_metrics():
    y_true = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    y_pred = np.array([2.1, 3.8, 6.2, 7.9, 9.8])

    metrics = compute_avian_metrics(y_true, y_pred, n_bootstrap=100)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mae_ci_95" in metrics
    assert "pearson_r" in metrics
    assert "spearman_rho" in metrics
    assert metrics["mae"] < 0.5
    assert metrics["r2"] > 0.9
