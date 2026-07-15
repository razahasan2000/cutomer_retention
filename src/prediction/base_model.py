from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class BaseModel(ABC):
    def __init__(self, name: str = "base", params: Optional[Dict] = None):
        self.name = name
        self.params = params or {}
        self.model = None
        self.fitted = False

    @abstractmethod
    def _build_model(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        self.model = self._build_model()
        self.model.fit(X, y)
        self.fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError(f"Model {self.name} not fitted yet.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError(f"Model {self.name} not fitted yet.")
        probs = self.model.predict_proba(X)
        return probs[:, 1] if probs.ndim == 2 and probs.shape[1] == 2 else probs

    def get_feature_importance(self) -> Optional[np.ndarray]:
        return None

    def get_params(self) -> Dict:
        return self.params
