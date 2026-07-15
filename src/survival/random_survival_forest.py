import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import Optional


class RandomSurvivalForestAnalysis:
    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.fitted = True
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("RSF not fitted.")
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> Optional[np.ndarray]:
        if self.fitted:
            return self.model.feature_importances_
        return None
