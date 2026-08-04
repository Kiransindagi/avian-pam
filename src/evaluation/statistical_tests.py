import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.config.schema import AppConfig
from src.utils.io import ensure_dir
from src.utils.logging import setup_logger

logger = setup_logger("StatisticalSignificanceTester")


class StatisticalSignificanceTester:
    """Hypothesis Testing Suite for Model Pairwise Performance Comparisons."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.reports_dir = ensure_dir(config.paths.reports_dir)

    def compare_models(
        self,
        y_true: np.ndarray,
        preds_a: np.ndarray,
        preds_b: np.ndarray,
        model_a_name: str = "Model_A",
        model_b_name: str = "Model_B",
        n_permutations: int = 1000,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Runs hypothesis tests comparing error residuals between Model A and Model B."""
        err_a = np.abs(y_true - preds_a)
        err_b = np.abs(y_true - preds_b)
        diff = err_a - err_b

        # 1. Paired t-test
        t_stat, t_p = stats.ttest_rel(err_a, err_b)

        # 2. Wilcoxon Signed-Rank Test
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(err_a, err_b)
        except Exception:
            wilcoxon_stat, wilcoxon_p = 0.0, 1.0

        # 3. Permutation Test on Mean Absolute Error Difference
        rng = np.random.RandomState(random_state)
        observed_diff = np.mean(diff)
        count_equal_or_more_extreme = 0

        for _ in range(n_permutations):
            swap = rng.rand(len(diff)) > 0.5
            perm_diff = np.where(swap, -diff, diff)
            if abs(np.mean(perm_diff)) >= abs(observed_diff):
                count_equal_or_more_extreme += 1

        permutation_p = count_equal_or_more_extreme / max(1, n_permutations)
        is_statistically_significant = bool(permutation_p < 0.05 or t_p < 0.05)

        return {
            "model_a": model_a_name,
            "model_b": model_b_name,
            "mae_a": round(float(np.mean(err_a)), 4),
            "mae_b": round(float(np.mean(err_b)), 4),
            "mae_diff": round(float(observed_diff), 4),
            "paired_t_stat": round(float(t_stat), 4),
            "paired_t_pvalue": round(float(t_p), 5),
            "wilcoxon_pvalue": round(float(wilcoxon_p), 5),
            "permutation_pvalue": round(float(permutation_p), 5),
            "is_significant_at_05": is_statistically_significant,
        }

    def generate_pairwise_significance_report(
        self,
        results_list: List[Dict[str, Any]],
    ) -> Path:
        """Generates statistical_tests.md report."""
        out_path = self.reports_dir / "statistical_tests.md"

        table_rows = ""
        for res in results_list:
            sig_badge = "✅ **Significant** ($p < 0.05$)" if res["is_significant_at_05"] else "❌ Not Significant"
            table_rows += (
                f"| `{res['model_a']}` vs `{res['model_b']}` | {res['mae_a']:.3f} | {res['mae_b']:.3f} | "
                f"{res['mae_diff']:.3f} | {res['paired_t_pvalue']:.4f} | {res['permutation_pvalue']:.4f} | {sig_badge} |\n"
            )

        content = f"""# Statistical Significance Testing Report

**Project**: {self.config.project.name}  
**Evaluation Protocol**: Paired t-test, Wilcoxon Signed-Rank Test & 1,000 Permutations.

---

## 1. Model Pairwise Comparison Matrix

| Comparison Pair | MAE Model A | MAE Model B | MAE Delta ($\Delta$) | Paired t-test $p$ | Permutation $p$ | Statistical Significance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{table_rows}

---

## 2. Methodology & Scientific Rigor
- **Null Hypothesis ($H_0$)**: There is no difference in prediction error between Model A and Model B.
- **Alternative Hypothesis ($H_1$)**: One model yields statistically significantly lower mean absolute error.
- **Significance Level ($\alpha$)**: $0.05$.
"""

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Saved statistical significance report to '{out_path}'.")
        return out_path
