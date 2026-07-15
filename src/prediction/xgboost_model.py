from xgboost import XGBClassifier
from src.prediction.base_model import BaseModel


class XGBoostModel(BaseModel):
    def __init__(self, params: dict = None):
        super().__init__(name="XGBoost", params=params)

    def _build_model(self):
        return XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1, **(self.params or {}))

    def get_feature_importance(self):
        if self.fitted:
            return self.model.feature_importances_
        return None
