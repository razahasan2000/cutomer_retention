import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from typing import Dict


class CalibrationEvaluator:
    @staticmethod
    def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        return float(brier_score_loss(y_true, y_prob))

    @staticmethod
    def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                                   n_bins: int = 10) -> Dict:
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        ece = 0.0
        bin_data = []
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            ece += bin_weight * abs(bin_acc - bin_conf)
            bin_data.append({
                "bin": i,
                "n_samples": int(mask.sum()),
                "accuracy": round(float(bin_acc), 4),
                "confidence": round(float(bin_conf), 4),
                "gap": round(float(abs(bin_acc - bin_conf)), 4)
            })
        return {
            "ece": round(ece, 4),
            "brier": CalibrationEvaluator.brier_score(y_true, y_prob),
            "bins": pd.DataFrame(bin_data)
        }
