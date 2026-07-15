import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from typing import Dict, Tuple


class FeatureAblation:
    def __init__(self, model=None, cv: int = 5, random_state: int = 42):
        self.model = model or LogisticRegression(max_iter=1000, random_state=random_state)
        self.cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        self.results: Dict[str, Dict] = {}

    def evaluate(self, X: pd.DataFrame, y: np.ndarray,
                 feature_groups: Dict[str, list]) -> pd.DataFrame:
        full_score = cross_val_score(self.model, X.values, y, cv=self.cv,
                                     scoring="roc_auc").mean()
        self.results["full"] = {"mean_auc": full_score, "std_auc": 0.0}

        for group_name, group_cols in feature_groups.items():
            cols_present = [c for c in group_cols if c in X.columns]
            if not cols_present:
                continue
            X_dropped = X.drop(columns=cols_present)
            scores = cross_val_score(self.model, X_dropped.values, y,
                                     cv=self.cv, scoring="roc_auc")
            self.results[group_name] = {
                "mean_auc": scores.mean(),
                "std_auc": scores.std(),
                "drop_features": cols_present,
                "delta": full_score - scores.mean()
            }

        rows = []
        for name, res in self.results.items():
            rows.append({
                "feature_group": name,
                "mean_auc": round(res["mean_auc"], 4),
                "std_auc": round(res.get("std_auc", 0), 4),
                "delta_from_full": round(res.get("delta", 0), 4),
                "dropped_features": str(res.get("drop_features", []))
            })
        return pd.DataFrame(rows)

    def evaluate_engineered_features(self, X_with: np.ndarray,
                                     X_without: np.ndarray,
                                     y: np.ndarray) -> Tuple[float, float]:
        auc_with = cross_val_score(self.model, X_with, y,
                                   cv=self.cv, scoring="roc_auc").mean()
        auc_without = cross_val_score(self.model, X_without, y,
                                      cv=self.cv, scoring="roc_auc").mean()
        return auc_with, auc_without
