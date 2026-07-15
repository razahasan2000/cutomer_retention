import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Any


class SHAPExplainer:
    def __init__(self, model: Any, X_sample: np.ndarray, model_type: str = "tree"):
        self.model = model
        self.X_sample = X_sample
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None

    def explain(self, X: Optional[np.ndarray] = None):
        X = X if X is not None else self.X_sample
        if self.model_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == "linear":
            self.explainer = shap.LinearExplainer(self.model, self.X_sample)
        else:
            self.explainer = shap.KernelExplainer(self.model.predict_proba, self.X_sample)
        self.shap_values = self.explainer.shap_values(X)
        return self

    def summary_plot(self, feature_names: list, max_display: int = 15, ax=None):
        shap.summary_plot(
            self.shap_values, self.X_sample, feature_names=feature_names,
            max_display=max_display, show=False
        )

    def dependence_plot(self, feature_idx: int, feature_names: list, ax=None):
        shap.dependence_plot(feature_idx, self.shap_values, self.X_sample,
                             feature_names=feature_names, show=False)

    def waterfall_plot(self, instance_idx: int, feature_names: list,
                       expected_value: Optional[float] = None):
        ev = expected_value if expected_value is not None else self.explainer.expected_value
        if isinstance(ev, np.ndarray):
            ev = ev[1] if ev.ndim > 0 else float(ev)
        shap.plots.waterfall(
            shap.Explanation(
                values=self.shap_values[instance_idx] if self.shap_values.ndim == 2
                else self.shap_values[1][instance_idx],
                base_values=ev,
                data=self.X_sample[instance_idx],
                feature_names=feature_names
            ),
            show=False
        )

    def global_importance(self, feature_names: list) -> pd.DataFrame:
        vals = np.abs(self.shap_values).mean(axis=0)
        if vals.ndim > 1 and vals.shape[1] > 1:
            vals = vals.mean(axis=0)
        return pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": vals
        }).sort_values("mean_abs_shap", ascending=False)
