import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from typing import Dict, Optional, List


class KaplanMeierAnalysis:
    def __init__(self):
        self.kmf = KaplanMeierFitter()
        self.results: Dict = {}

    def fit(self, durations: np.ndarray, event_observed: np.ndarray, label: str = "All"):
        self.kmf.fit(durations, event_observed, label=label)
        self.results[label] = {
            "median_survival_time": self.kmf.median_survival_time_,
            "survival_function": self.kmf.survival_function_,
        }
        return self

    def plot_survival_curve(self, ax=None, **kwargs):
        return self.kmf.plot_survival_function(ax=ax, **kwargs)

    def compare_groups(self, durations: np.ndarray, event: np.ndarray,
                       groups: np.ndarray, group_names: List[str]) -> pd.DataFrame:
        results = []
        for i, g1 in enumerate(group_names):
            for g2 in group_names[i + 1:]:
                mask1 = groups == g1
                mask2 = groups == g2
                if mask1.sum() < 5 or mask2.sum() < 5:
                    continue
                result = logrank_test(
                    durations[mask1 | mask2],
                    durations[mask1 | mask2],
                    event_observed_A=event[mask1],
                    event_observed_B=event[mask2]
                )
                results.append({
                    "group_1": g1,
                    "group_2": g2,
                    "test_statistic": round(result.test_statistic, 4),
                    "p_value": round(result.p_value, 6),
                })
        return pd.DataFrame(results)
