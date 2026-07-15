import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict


def plot_model_comparison(metrics_df: pd.DataFrame, metric_col: str = "roc_auc",
                          title: str = "Model Comparison",
                          save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    models = metrics_df.index.tolist()
    values = metrics_df[metric_col].values
    colors = ["#2e86c1", "#1abc9c", "#e67e22", "#e74c3c", "#8e44ad", "#2c3e50", "#16a085"]
    bars = ax.barh(range(len(models)), values, color=colors[:len(models)], edgecolor="white")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel(metric_col.upper(), fontsize=11)
    ax.set_title(title, fontsize=12)
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.set_xlim(0, max(values) * 1.15)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_cross_dataset_heatmap(pivot_df: pd.DataFrame, title: str = "Cross-Dataset AUC",
                               save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot_df.values, cmap="YlOrRd", aspect="auto", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index)
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            ax.text(j, i, f"{pivot_df.values[i, j]:.3f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if pivot_df.values[i, j] > 0.7 else "black")
    fig.colorbar(im, ax=ax, label="ROC AUC")
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
