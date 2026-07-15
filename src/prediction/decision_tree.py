from sklearn.tree import DecisionTreeClassifier
from src.prediction.base_model import BaseModel


class DecisionTreeModel(BaseModel):
    def __init__(self, params: dict = None):
        super().__init__(name="Decision Tree", params=params)

    def _build_model(self):
        return DecisionTreeClassifier(max_depth=6, random_state=42, **(self.params or {}))

    def get_feature_importance(self):
        if self.fitted:
            return self.model.feature_importances_
        return None
