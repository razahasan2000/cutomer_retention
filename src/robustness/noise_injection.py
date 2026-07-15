import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Dict, Any, Callable


class NoiseInjector:
    def __init__(self, model: Any, cv: int = 5, random_state: int = 42):
        self.model = model
        self.cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        self.random_state = random_state
        self.results: Dict = {}

    def gaussian_noise(self, X: np.ndarray, y: np.ndarray,
                       noise_levels: list = None) -> pd.DataFrame:
        noise_levels = noise_levels or [0.01, 0.05, 0.1, 0.2]
        rows = []
        base_score = cross_val_score(self.model, X, y, cv=self.cv, scoring="roc_auc").mean()
        rows.append({"noise_level": 0.0, "mean_auc": round(base_score, 4), "delta": 0.0})
        for sigma in noise_levels:
            rng = np.random.RandomState(self.random_state)
            X_noisy = X + rng.randn(*X.shape) * sigma
            score = cross_val_score(self.model, X_noisy, y, cv=self.cv, scoring="roc_auc").mean()
            rows.append({
                "noise_level": sigma,
                "mean_auc": round(score, 4),
                "delta": round(base_score - score, 4)
            })
        self.results["gaussian"] = rows
        return pd.DataFrame(rows)

    def label_noise(self, X: np.ndarray, y: np.ndarray,
                    flip_rates: list = None) -> pd.DataFrame:
        flip_rates = flip_rates or [0.01, 0.03, 0.05, 0.1]
        rows = []
        base_score = cross_val_score(self.model, X, y, cv=self.cv, scoring="roc_auc").mean()
        rows.append({"flip_rate": 0.0, "mean_auc": round(base_score, 4), "delta": 0.0})
        rng = np.random.RandomState(self.random_state)
        for rate in flip_rates:
            y_noisy = y.copy()
            n_flip = int(len(y) * rate)
            flip_idx = rng.choice(len(y), n_flip, replace=False)
            y_noisy[flip_idx] = 1 - y_noisy[flip_idx]
            score = cross_val_score(self.model, X, y_noisy, cv=self.cv, scoring="roc_auc").mean()
            rows.append({
                "flip_rate": rate,
                "mean_auc": round(score, 4),
                "delta": round(base_score - score, 4)
            })
        self.results["label"] = rows
        return pd.DataFrame(rows)

    def missing_data(self, X: np.ndarray, y: np.ndarray,
                     missing_rates: list = None) -> pd.DataFrame:
        missing_rates = missing_rates or [0.05, 0.1, 0.2, 0.3]
        rows = []
        base_score = cross_val_score(self.model, X, y, cv=self.cv, scoring="roc_auc").mean()
        rows.append({"missing_rate": 0.0, "mean_auc": round(base_score, 4), "delta": 0.0})
        rng = np.random.RandomState(self.random_state)
        for rate in missing_rates:
            X_missing = X.copy()
            mask = rng.binomial(1, rate, X.shape).astype(bool)
            X_missing[mask] = 0
            score = cross_val_score(self.model, X_missing, y, cv=self.cv, scoring="roc_auc").mean()
            rows.append({
                "missing_rate": rate,
                "mean_auc": round(score, 4),
                "delta": round(base_score - score, 4)
            })
        self.results["missing"] = rows
        return pd.DataFrame(rows)
