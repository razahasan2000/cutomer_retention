import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from typing import Dict, List


class ExplanationEvaluation:
    @staticmethod
    def rank_agreement(rankings_a: List[str], rankings_b: List[str]) -> Dict:
        common = list(set(rankings_a) & set(rankings_b))
        if len(common) < 3:
            return {"spearman": 0, "kendall": 0, "common_features": len(common)}
        ranks_a = [rankings_a.index(f) if f in rankings_a else len(rankings_a) for f in common]
        ranks_b = [rankings_b.index(f) if f in rankings_b else len(rankings_b) for f in common]
        sp, _ = spearmanr(ranks_a, ranks_b)
        kt, _ = kendalltau(ranks_a, ranks_b)
        return {
            "spearman": round(sp, 4) if not np.isnan(sp) else 0,
            "kendall": round(kt, 4) if not np.isnan(kt) else 0,
            "common_features": len(common)
        }

    @staticmethod
    def stability_across_runs(shap_values_list: List[np.ndarray],
                              feature_names: List[str]) -> pd.DataFrame:
        n_features = len(feature_names)
        importance_matrix = np.zeros((len(shap_values_list), n_features))
        for i, sv in enumerate(shap_values_list):
            if sv.ndim > 1 and sv.shape[1] > 1:
                sv = sv.mean(axis=0)
            importance_matrix[i] = np.abs(sv).mean(axis=0) if sv.ndim == 2 else np.abs(sv)
        stds = importance_matrix.std(axis=0)
        means = importance_matrix.mean(axis=0)
        cv = stds / (means + 1e-10)
        return pd.DataFrame({
            "feature": feature_names,
            "mean_importance": means,
            "std_importance": stds,
            "cv": cv
        }).sort_values("mean_importance", ascending=False)
