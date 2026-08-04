import numpy as np

from src.config.schema import AppConfig
from src.evaluation.statistical_tests import StatisticalSignificanceTester


def test_statistical_significance_tester(tmp_path):
    config = AppConfig()
    config.paths.reports_dir = str(tmp_path)
    tester = StatisticalSignificanceTester(config)

    y_true = np.array([2, 4, 6, 8, 10, 12, 14, 16])
    preds_a = np.array([2.1, 4.0, 6.1, 7.9, 10.1, 11.9, 14.1, 15.9])  # Accurate
    preds_b = np.array([5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0])  # Inaccurate

    res = tester.compare_models(
        y_true, preds_a, preds_b, "Model_A", "Model_B", n_permutations=200
    )

    assert res["mae_a"] < res["mae_b"]
    assert "paired_t_pvalue" in res
    assert "permutation_pvalue" in res
