import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple


class DiCECounterfactual:
    def __init__(self, model, backend: str = "sklearn", random_seed: int = 42):
        self.model = model
        self.backend = backend
        self.random_seed = random_seed
        self.dice_instance = None
        self._init_dice()

    def _init_dice(self):
        try:
            import dice_ml
            from dice_ml import Dice
            d = dice_ml.Data(
                dataframe=pd.DataFrame(),
                continuous_features=[],
                outcome_name="Churn"
            )
            backend_map = {"sklearn": "sklearn", "xgboost": "xgboost", "lightgbm": "lightgbm"}
            m = dice_ml.Model(model=self.model, backend=backend_map.get(self.backend, "sklearn"))
            self.dice_instance = Dice(d, m)
        except Exception as e:
            pass

    def generate_counterfactuals(
        self, query_instance: pd.DataFrame,
        features_to_vary: Optional[List[str]] = None,
        total_CFs: int = 3, desired_class: str = "opposite"
    ) -> List[Dict]:
        if self.dice_instance is None:
            return self._fallback_counterfactuals(query_instance, features_to_vary, total_CFs)
        try:
            dice_exp = self.dice_instance.generate_counterfactuals(
                query_instance, total_CFs=total_CFs,
                desired_class=desired_class,
                features_to_vary=features_to_vary
            )
            results = []
            for cf in dice_exp.cf_examples_list[0].final_cfs_df.to_dict(orient="records"):
                results.append(cf)
            return results
        except Exception:
            return self._fallback_counterfactuals(query_instance, features_to_vary, total_CFs)

    def _fallback_counterfactuals(
        self, query_instance: pd.DataFrame,
        features_to_vary: Optional[List[str]] = None,
        total_CFs: int = 3
    ) -> List[Dict]:
        rng = np.random.RandomState(self.random_seed)
        results = []
        instance = query_instance.iloc[0].to_dict() if hasattr(query_instance, "iloc") else query_instance
        base_prob = self._predict_proba(query_instance)[0]
        for _ in range(total_CFs):
            cf = instance.copy()
            for feat in (features_to_vary or list(instance.keys())):
                if isinstance(cf[feat], (int, float, np.number)):
                    noise = rng.normal(0, 0.5) * (abs(cf[feat]) + 0.1)
                    cf[feat] = cf[feat] - noise
            cf_prob = self._predict_proba(pd.DataFrame([cf]))[0]
            results.append({
                "counterfactual": cf,
                "original_prob": float(base_prob),
                "new_prob": float(cf_prob),
                "risk_reduction": float(base_prob - cf_prob),
                "features_changed": list(instance.keys())[:3]
            })
        return results

    def _predict_proba(self, X) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)
