import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix
)
from typing import Dict, Tuple


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
    }


def classification_report_df(
    models: Dict[str, Tuple[np.ndarray, np.ndarray]],
    y_true: np.ndarray
) -> pd.DataFrame:
    rows = []
    for name, (y_pred, y_prob) in models.items():
        m = calculate_metrics(y_true, y_pred, y_prob)
        m["model"] = name
        rows.append(m)
    return pd.DataFrame(rows).set_index("model")


def bootstrap_ci(
    y_true: np.ndarray, y_prob: np.ndarray,
    metric_fn: callable, n_bootstrap: int = 1000, alpha: float = 0.05
) -> Tuple[float, float, float]:
    rng = np.random.RandomState(42)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_prob[idx]))
    scores = np.array(scores)
    return float(np.mean(scores)), float(np.percentile(scores, 100 * alpha / 2)), float(np.percentile(scores, 100 * (1 - alpha / 2)))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_prob))


def calibration_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    rows = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": i,
            "bin_mid": (bins[i] + bins[i + 1]) / 2,
            "n_samples": int(mask.sum()),
            "mean_predicted": float(y_prob[mask].mean()),
            "fraction_positives": float(y_true[mask].mean()),
        })
    return pd.DataFrame(rows)
