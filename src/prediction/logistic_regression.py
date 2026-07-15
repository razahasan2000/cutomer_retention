from sklearn.linear_model import LogisticRegression
from src.prediction.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(self, params: dict = None):
        super().__init__(name="Logistic Regression", params=params)

    def _build_model(self):
        return LogisticRegression(max_iter=1000, random_state=42, **(self.params or {}))

    def get_feature_importance(self):
        if self.fitted:
            return self.model.coef_[0]
        return None
