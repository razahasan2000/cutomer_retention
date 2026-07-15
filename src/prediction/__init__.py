from src.prediction.base_model import BaseModel
from src.prediction.logistic_regression import LogisticRegressionModel
from src.prediction.decision_tree import DecisionTreeModel
from src.prediction.random_forest import RandomForestModel
from src.prediction.xgboost_model import XGBoostModel
from src.prediction.lightgbm_model import LightGBMModel
from src.prediction.catboost_model import CatBoostModel
from src.prediction.tabnet_model import TabNetModel
from src.prediction.ensemble import EnsembleModel

MODEL_REGISTRY = {
    "Logistic Regression": LogisticRegressionModel,
    "Decision Tree": DecisionTreeModel,
    "Random Forest": RandomForestModel,
    "XGBoost": XGBoostModel,
    "LightGBM": LightGBMModel,
    "CatBoost": CatBoostModel,
    "TabNet": TabNetModel,
}
