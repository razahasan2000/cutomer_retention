import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from src.prediction.base_model import BaseModel


class TabNetModel(BaseModel):
    def __init__(self, params: dict = None, epochs: int = 30, patience: int = 10, batch_size: int = 256):
        super().__init__(name="TabNet", params=params)
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size

    def _build_model(self):
        p = self.params or {}
        return TabNetClassifier(
            n_d=p.get("n_d", 8),
            n_a=p.get("n_a", 8),
            n_steps=p.get("n_steps", 3),
            gamma=p.get("gamma", 1.3),
            lambda_sparse=p.get("lambda_sparse", 0.001),
            optimizer_fn=torch.optim.Adam,
            optimizer_params=p.get("optimizer_params", {"lr": 2e-2}),
            scheduler_params=p.get("scheduler_params", {"step_size": 50, "gamma": 0.9}),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            epsilon=p.get("epsilon", 1e-15),
            seed=p.get("seed", 42),
            verbose=0
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabNetModel":
        self.model = self._build_model()
        y = y.ravel()
        self.model.fit(
            X_train=X, y_train=y,
            eval_set=[(X, y)],
            patience=self.patience,
            max_epochs=self.epochs,
            batch_size=self.batch_size,
            virtual_batch_size=128
        )
        self.fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TabNet not fitted yet.")
        return self.model.predict(X).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TabNet not fitted yet.")
        probs = self.model.predict_proba(X)
        return probs[:, 1] if probs.ndim == 2 and probs.shape[1] == 2 else probs.ravel()

    def get_feature_importance(self):
        if self.fitted:
            return self.model.feature_importances_
        return None
