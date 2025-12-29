"""
Meridian Stats - Significance Testing
"""

import numpy as np
from scipy import stats
from typing import Optional


def paired_ttest(
    values_a: list[float],
    values_b: list[float],
) -> dict:
    """Paired t-test for dependent samples."""
    t_stat, p_value = stats.ttest_rel(values_a, values_b)
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_05": p_value < 0.05,
        "significant_01": p_value < 0.01,
    }


def mcnemar_test(
    a_correct_b_wrong: int,
    a_wrong_b_correct: int,
) -> dict:
    """McNemar's test for comparing classifiers."""
    # With continuity correction
    n = a_correct_b_wrong + a_wrong_b_correct
    if n == 0:
        return {"chi2": 0, "p_value": 1.0, "significant": False}
    
    chi2 = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1) ** 2 / n
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "significant_05": p_value < 0.05,
    }


def cohens_d(
    values_a: list[float],
    values_b: list[float],
) -> float:
    """Calculate Cohen's d effect size."""
    a, b = np.array(values_a), np.array(values_b)
    n_a, n_b = len(a), len(b)
    
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return float((np.mean(b) - np.mean(a)) / pooled_std)
