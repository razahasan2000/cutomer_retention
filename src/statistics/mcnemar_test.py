import numpy as np
from scipy.stats import chi2
from typing import Dict


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray,
                 y_pred_b: np.ndarray) -> Dict:
    n00 = np.sum((y_pred_a == 0) & (y_pred_b == 0))
    n01 = np.sum((y_pred_a == 0) & (y_pred_b == 1))
    n10 = np.sum((y_pred_a == 1) & (y_pred_b == 0))
    n11 = np.sum((y_pred_a == 1) & (y_pred_b == 1))
    b = n01
    c = n10
    numerator = (abs(b - c) - 1) ** 2
    denominator = b + c
    if denominator == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n_discordant": 0}
    chi2_stat = numerator / denominator
    p = 1 - chi2.cdf(chi2_stat, df=1)
    return {
        "chi2": round(chi2_stat, 4),
        "p_value": round(p, 6),
        "n_discordant": int(denominator),
        "significant": p < 0.05
    }
