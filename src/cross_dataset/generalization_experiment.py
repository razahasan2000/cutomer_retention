import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from typing import Dict, List, Any, Tuple


class CrossDatasetExperiment:
    def __init__(self):
        self.results: List[Dict] = []

    def run_experiment(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       train_name: str, test_name: str,
                       model_name: str) -> Dict:
        if X_train.shape[1] != X_test.shape[1]:
            result = {
                "train_dataset": train_name,
                "test_dataset": test_name,
                "model": model_name,
                "roc_auc": -1.0,
                "pr_auc": -1.0,
                "brier": -1.0,
                "note": f"Feature mismatch: {X_train.shape[1]} != {X_test.shape[1]}"
            }
            self.results.append(result)
            return result
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                y_prob = y_prob[:, 1]
        else:
            y_prob = model.predict(X_test)
        y_pred = (y_prob >= 0.5).astype(int)
        result = {
            "train_dataset": train_name,
            "test_dataset": test_name,
            "model": model_name,
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
            "pr_auc": round(average_precision_score(y_test, y_prob), 4),
            "brier": round(brier_score_loss(y_test, y_prob), 4),
        }
        self.results.append(result)
        return result

    def run_all_transfer_experiments(self, models: Dict[str, Any],
                                     datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                     verbose: bool = True) -> pd.DataFrame:
        for model_name, model in models.items():
            for train_name, (X_tr, y_tr) in datasets.items():
                for test_name, (X_te, y_te) in datasets.items():
                    if train_name == test_name:
                        continue
                    if verbose:
                        print(f"  {model_name}: {train_name} -> {test_name}")
                    self.run_experiment(model, X_tr, y_tr, X_te, y_te,
                                        train_name, test_name, model_name)
        return pd.DataFrame(self.results)

    def summary_table(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame({"info": ["No cross-dataset results. Models may have failed to train."]})
        df = pd.DataFrame(self.results)
        if "roc_auc" not in df.columns:
            return pd.DataFrame({"info": ["No roc_auc column in results."]})
        pivot = df.pivot_table(
            index=["train_dataset", "test_dataset"],
            columns="model",
            values="roc_auc",
            aggfunc="first"
        )
        pivot["mean_auc"] = pivot.mean(axis=1)
        pivot["std_auc"] = pivot.std(axis=1)
        return pivot.round(4)
