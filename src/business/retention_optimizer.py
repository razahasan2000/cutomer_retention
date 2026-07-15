import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class RetentionOptimizer:
    def __init__(self, clv_model=None):
        self.clv_model = clv_model

    def simulate_intervention(self, base_risk: float, new_risk: float,
                              clv: float, intervention_cost: float) -> Dict:
        risk_reduction = base_risk - new_risk
        expected_revenue_saved = risk_reduction * clv
        net_benefit = expected_revenue_saved - intervention_cost
        roi = (net_benefit / intervention_cost * 100) if intervention_cost > 0 else 0
        return {
            "base_risk": round(base_risk, 4),
            "new_risk": round(new_risk, 4),
            "risk_reduction": round(risk_reduction, 4),
            "clv": round(clv, 2),
            "intervention_cost": round(intervention_cost, 2),
            "expected_revenue_saved": round(expected_revenue_saved, 2),
            "net_benefit": round(net_benefit, 2),
            "roi_pct": round(roi, 2),
            "recommendation": self._get_recommendation(roi, risk_reduction)
        }

    def optimize_intervention(self, customer_data: pd.DataFrame,
                              churn_probs: np.ndarray,
                              clv_values: np.ndarray,
                              intervention_costs: List[float],
                              risk_thresholds: List[float]) -> pd.DataFrame:
        results = []
        for cost in intervention_costs:
            for threshold in risk_thresholds:
                at_risk = churn_probs >= threshold
                n_at_risk = at_risk.sum()
                if n_at_risk == 0:
                    continue
                total_cost = n_at_risk * cost
                avg_clv = clv_values[at_risk].mean()
                total_value_saved = avg_clv * n_at_risk * 0.3
                net = total_value_saved - total_cost
                roi = (net / total_cost * 100) if total_cost > 0 else 0
                results.append({
                    "risk_threshold": threshold,
                    "intervention_cost": cost,
                    "n_at_risk": n_at_risk,
                    "total_intervention_cost": round(total_cost, 2),
                    "total_value_saved": round(total_value_saved, 2),
                    "net_benefit": round(net, 2),
                    "roi_pct": round(roi, 2)
                })
        return pd.DataFrame(results).sort_values("roi_pct", ascending=False)

    def _get_recommendation(self, roi: float, risk_reduction: float) -> str:
        if roi > 200:
            return "Highly recommended - strong ROI"
        elif roi > 50:
            return "Recommended - positive ROI"
        elif roi > 0:
            return "Marginally beneficial"
        else:
            return "Not recommended - negative ROI"
