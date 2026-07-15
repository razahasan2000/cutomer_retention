from catboost import CatBoostClassifier
from src.prediction.base_model import BaseModel


class CatBoostModel(BaseModel):
    def __init__(self, params: dict = None):
        super().__init__(name="CatBoost", params=params)

    def _build_model(self):
        return CatBoostClassifier(random_state=42, verbose=0, **(self.params or {}))

    def get_feature_importance(self):
        if self.fitted:
            return self.model.get_feature_importance()
        return None
