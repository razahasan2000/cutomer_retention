import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from typing import Any


class PermutationImportance:
    def __init__(self, model: Any, X: np.ndarray, y: np.ndarray,
                 n_repeats: int = 10, random_state: int = 42):
        self.model = model
        self.X = X
        self.y = y
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.result = None

    def compute(self):
        self.result = permutation_importance(
            self.model, self.X, self.y,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            n_jobs=-1
        )
        return self

    def as_dataframe(self, feature_names: list) -> pd.DataFrame:
        if self.result is None:
            self.compute()
        return pd.DataFrame({
            "feature": feature_names,
            "importance_mean": self.result.importances_mean,
            "importance_std": self.result.importances_std
        }).sort_values("importance_mean", ascending=False)
