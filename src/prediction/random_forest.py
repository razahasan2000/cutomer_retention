from sklearn.ensemble import RandomForestClassifier
from src.prediction.base_model import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, params: dict = None):
        super().__init__(name="Random Forest", params=params)

    def _build_model(self):
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, **(self.params or {}))

    def get_feature_importance(self):
        if self.fitted:
            return self.model.feature_importances_
        return None
