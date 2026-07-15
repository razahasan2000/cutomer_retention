import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from typing import Dict, List


def plot_roc_curves(y_true: np.ndarray, model_probs: Dict[str, np.ndarray],
                    title: str = "ROC Curves", save_path: str = None):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2e86c1", "#1abc9c", "#e67e22", "#e74c3c", "#8e44ad", "#2c3e50", "#16a085"]
    for i, (name, y_prob) in enumerate(model_probs.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color=colors[i % len(colors)],
                linewidth=2, label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_calibration_curves(y_true: np.ndarray, model_probs: Dict[str, np.ndarray],
                            title: str = "Calibration Curves", n_bins: int = 10,
                            save_path: str = None):
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2e86c1", "#1abc9c", "#e67e22", "#e74c3c", "#8e44ad", "#2c3e50", "#16a085"]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfectly Calibrated")
    for i, (name, y_prob) in enumerate(model_probs.items()):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        ax.plot(prob_pred, prob_true, marker="o", color=colors[i % len(colors)],
                linewidth=2, label=name)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
