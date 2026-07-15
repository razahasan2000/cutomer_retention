import matplotlib.pyplot as plt
import shap
import numpy as np
from typing import List


def plot_shap_summary(shap_values: np.ndarray, X: np.ndarray,
                      feature_names: List[str], max_display: int = 15,
                      title: str = "SHAP Summary Plot", save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      max_display=max_display, show=False)
    plt.title(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_shap_dependence(feature_idx: str, shap_values: np.ndarray,
                         X: np.ndarray, feature_names: List[str],
                         title: str = None, save_path: str = None):
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.dependence_plot(feature_idx, shap_values, X,
                         feature_names=feature_names, show=False, ax=ax)
    if title:
        ax.set_title(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_shap_waterfall(shap_values: shap.Explanation, instance_idx: int = 0,
                        max_display: int = 10, save_path: str = None):
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[instance_idx], max_display=max_display, show=False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
