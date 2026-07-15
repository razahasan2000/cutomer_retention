import numpy as np
import pandas as pd
from typing import Dict, List


class FairnessEvaluator:
    def __init__(self):
        self.results: Dict = {}

    def demographic_parity(self, y_pred: np.ndarray,
                           sensitive_attr: np.ndarray,
                           attr_name: str) -> pd.DataFrame:
        groups = np.unique(sensitive_attr)
        rows = []
        overall_rate = y_pred.mean()
        for g in groups:
            mask = sensitive_attr == g
            group_rate = y_pred[mask].mean()
            rows.append({
                "attribute": attr_name,
                "group": str(g),
                "n": int(mask.sum()),
                "positive_rate": round(float(group_rate), 4),
                "overall_rate": round(float(overall_rate), 4),
                "disparity": round(float(group_rate - overall_rate), 4)
            })
        return pd.DataFrame(rows)

    def equal_opportunity(self, y_true: np.ndarray, y_pred: np.ndarray,
                          sensitive_attr: np.ndarray,
                          attr_name: str) -> pd.DataFrame:
        groups = np.unique(sensitive_attr)
        rows = []
        for g in groups:
            mask = (sensitive_attr == g) & (y_true == 1)
            if mask.sum() == 0:
                continue
            tpr = y_pred[mask].mean()
            rows.append({
                "attribute": attr_name,
                "group": str(g),
                "n_positive": int(mask.sum()),
                "true_positive_rate": round(float(tpr), 4)
            })
        return pd.DataFrame(rows)

    def equalized_odds(self, y_true: np.ndarray, y_pred: np.ndarray,
                       sensitive_attr: np.ndarray,
                       attr_name: str) -> pd.DataFrame:
        groups = np.unique(sensitive_attr)
        rows = []
        for g in groups:
            for cls in [0, 1]:
                mask = (sensitive_attr == g) & (y_true == cls)
                if mask.sum() == 0:
                    continue
                rate = y_pred[mask].mean()
                rows.append({
                    "attribute": attr_name,
                    "group": str(g),
                    "class": cls,
                    "n": int(mask.sum()),
                    "prediction_rate": round(float(rate), 4)
                })
        return pd.DataFrame(rows)

    def evaluate_all(self, y_true: np.ndarray, y_pred: np.ndarray,
                     sensitive_df: pd.DataFrame) -> pd.DataFrame:
        all_rows = []
        for col in sensitive_df.columns:
            attr = sensitive_df[col].values
            dp = self.demographic_parity(y_pred, attr, col)
            eo = self.equal_opportunity(y_true, y_pred, attr, col)
            dp["metric"] = "demographic_parity"
            eo["metric"] = "equal_opportunity"
            all_rows.append(dp)
            all_rows.append(eo)
        return pd.concat(all_rows, ignore_index=True)
