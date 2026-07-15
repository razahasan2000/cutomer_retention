import numpy as np
from scipy.stats import norm
from typing import Tuple


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    n = len(x)
    t = x.argsort()
    y = x[t]
    z = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and y[j] == y[i]:
            j += 1
        z[t[i:j]] = (i + j - 1) / 2.0
        i = j
    return z


def _delong_covariance(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    n = len(y_true)
    idx_pos = y_true == 1
    idx_neg = y_true == 0
    n_pos = idx_pos.sum()
    n_neg = idx_neg.sum()
    if n_pos == 0 or n_neg == 0:
        return 0.0
    V10 = _compute_midrank(y_prob[idx_pos])
    V01 = _compute_midrank(y_prob[idx_neg])
    S10 = np.var(V10, ddof=1)
    S01 = np.var(V01, ddof=1)
    return S10 / n_pos + S01 / n_neg


def delong_roc_test(y_true: np.ndarray, y_prob_a: np.ndarray,
                    y_prob_b: np.ndarray) -> Tuple[float, float]:
    auc_a = np.mean(y_prob_a[y_true == 1] > y_prob_a[y_true == 0][:, None])
    auc_b = np.mean(y_prob_b[y_true == 1] > y_prob_b[y_true == 0][:, None])
    var_a = _delong_covariance(y_true, y_prob_a)
    var_b = _delong_covariance(y_true, y_prob_b)

    n = len(y_true)
    idx_pos = y_true == 1
    idx_neg = y_true == 0
    V10_a = _compute_midrank(y_prob_a[idx_pos])
    V01_a = _compute_midrank(y_prob_a[idx_neg])
    V10_b = _compute_midrank(y_prob_b[idx_pos])
    V01_b = _compute_midrank(y_prob_b[idx_neg])
    cov = np.cov(V10_a, V10_b)[0, 1] / idx_pos.sum() + np.cov(V01_a, V01_b)[0, 1] / idx_neg.sum()

    se = np.sqrt(var_a + var_b - 2 * cov)
    z = (auc_a - auc_b) / (se + 1e-10)
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p)
