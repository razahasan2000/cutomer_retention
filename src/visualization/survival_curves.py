import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from typing import Dict, List


def plot_kaplan_meier(durations: np.ndarray, event: np.ndarray,
                      groups: np.ndarray = None, group_names: List[str] = None,
                      title: str = "Kaplan-Meier Survival Curves",
                      save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    if groups is not None and group_names is not None:
        colors = ["#2e86c1", "#e74c3c", "#1abc9c", "#e67e22", "#8e44ad"]
        for i, name in enumerate(group_names):
            mask = groups == name
            if mask.sum() < 5:
                continue
            kmf.fit(durations[mask], event[mask], label=name)
            kmf.plot_survival_function(ax=ax, color=colors[i % len(colors)],
                                       linewidth=2, ci_show=True)
    else:
        kmf.fit(durations, event, label="All Customers")
        kmf.plot_survival_function(ax=ax, color="#2e86c1", linewidth=2, ci_show=True)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival Probability")
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_hazard_rates(durations: np.ndarray, event: np.ndarray,
                      title: str = "Hazard Function",
                      save_path: str = None):
    from lifelines import NelsonAalenFitter
    fig, ax = plt.subplots(figsize=(10, 6))
    naf = NelsonAalenFitter()
    naf.fit(durations, event, label="Cumulative Hazard")
    naf.plot(ax=ax, color="#e74c3c", linewidth=2)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Cumulative Hazard")
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
