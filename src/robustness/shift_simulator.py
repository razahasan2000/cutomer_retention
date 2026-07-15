import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Any, Dict


class ShiftSimulator:
    def __init__(self, model: Any, cv: int = 5, random_state: int = 42):
        self.model = model
        self.cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    def covariate_shift(self, X_source: np.ndarray, y_source: np.ndarray,
                        X_target: np.ndarray, y_target: np.ndarray) -> Dict:
        train_score = cross_val_score(self.model, X_source, y_source,
                                      cv=self.cv, scoring="roc_auc").mean()
        self.model.fit(X_source, y_source)
        if hasattr(self.model, "predict_proba"):
            y_prob = self.model.predict_proba(X_target)[:, 1]
        else:
            y_prob = self.model.predict(X_target)
        from sklearn.metrics import roc_auc_score
        test_score = roc_auc_score(y_target, y_prob)
        return {
            "train_auc": round(train_score, 4),
            "test_auc": round(test_score, 4),
            "degradation": round(train_score - test_score, 4)
        }
