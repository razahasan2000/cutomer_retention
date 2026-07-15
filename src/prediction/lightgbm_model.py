from lightgbm import LGBMClassifier
from src.prediction.base_model import BaseModel


class LightGBMModel(BaseModel):
    def __init__(self, params: dict = None):
        super().__init__(name="LightGBM", params=params)

    def _build_model(self):
        return LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **(self.params or {}))

    def get_feature_importance(self):
        if self.fitted:
            return self.model.feature_importances_
        return None
