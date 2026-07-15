import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from typing import Dict, Callable


class BootstrapInference:
    def __init__(self, n_bootstrap: int = 1000, alpha: float = 0.05, random_state: int = 42):
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.random_state = random_state

    def compute_ci(self, y_true: np.ndarray, y_prob: np.ndarray,
                   metric_fn: Callable = roc_auc_score) -> Dict:
        rng = np.random.RandomState(self.random_state)
        n = len(y_true)
        scores = []
        for _ in range(self.n_bootstrap):
            idx = rng.randint(0, n, n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            scores.append(metric_fn(y_true[idx], y_prob[idx]))
        scores = np.array(scores)
        return {
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "ci_lower": float(np.percentile(scores, 100 * self.alpha / 2)),
            "ci_upper": float(np.percentile(scores, 100 * (1 - self.alpha / 2))),
            "std": float(np.std(scores)),
            "n_bootstrap": len(scores)
        }

    def model_comparison(self, y_true: np.ndarray, y_prob_a: np.ndarray,
                         y_prob_b: np.ndarray) -> Dict:
        rng = np.random.RandomState(self.random_state)
        n = len(y_true)
        diffs = []
        for _ in range(self.n_bootstrap):
            idx = rng.randint(0, n, n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            auc_a = roc_auc_score(y_true[idx], y_prob_a[idx])
            auc_b = roc_auc_score(y_true[idx], y_prob_b[idx])
            diffs.append(auc_a - auc_b)
        diffs = np.array(diffs)
        p_value = float(np.mean(diffs <= 0))
        return {
            "mean_diff": float(np.mean(diffs)),
            "ci_lower": float(np.percentile(diffs, 100 * self.alpha / 2)),
            "ci_upper": float(np.percentile(diffs, 100 * (1 - self.alpha / 2))),
            "p_value": round(p_value, 6),
            "significant": p_value < self.alpha
        }
