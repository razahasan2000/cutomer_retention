import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from src.prediction.base_model import BaseModel


class EnsembleModel(BaseModel):
    def __init__(self, estimators: list, meta_model=None, params: dict = None):
        super().__init__(name="Ensemble (Stacking)", params=params)
        self.estimators = estimators
        self.meta_model = meta_model or LogisticRegression(max_iter=1000, random_state=42)

    def _build_model(self):
        return StackingClassifier(
            estimators=self.estimators,
            final_estimator=self.meta_model,
            cv=5,
            n_jobs=-1
        )

    def get_feature_importance(self):
        return None
