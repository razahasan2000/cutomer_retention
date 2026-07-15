import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def expected_profit(churn_probs: np.ndarray, clv_values: np.ndarray,
                    mask: np.ndarray, intervention_cost: float,
                    success_rate: float) -> Tuple[float, float, float, float]:
    """Principled expected-profit for a targeted subset defined by ``mask``.

    For each targeted customer ``i`` with churn probability ``p_i`` and CLV ``c_i``,
    expected revenue saved = success_rate * p_i * c_i. The total cost is the number
    of targeted customers times the per-customer intervention cost. Returns
    ``(expected_revenue_saved, total_cost, net_benefit, roi_pct)``.
    """
    idx = np.where(np.asarray(mask, dtype=bool))[0]
    if len(idx) == 0:
        return 0.0, 0.0, 0.0, 0.0
    expected_saved = float(np.sum(success_rate * churn_probs[idx] * clv_values[idx]))
    total_cost = float(len(idx) * intervention_cost)
    net = expected_saved - total_cost
    roi = (net / total_cost * 100.0) if total_cost > 0 else 0.0
    return expected_saved, total_cost, net, roi


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

    def optimal_threshold(self, churn_probs: np.ndarray, clv_values: np.ndarray,
                          intervention_cost: float, success_rate: float = 0.3,
                          budget: Optional[float] = None,
                          thresholds: Optional[List[float]] = None) -> Tuple[pd.DataFrame, Dict]:
        """Learn the intervention threshold that maximizes expected business profit.

        Profit is driven by the value-weighted *expected-profit score*
        ``s_i = success_rate * p_i * CLV_i`` (the same quantity the ESACRIF
        value-weighted strategy ranks by), not by the raw churn probability alone.
        This method sweeps candidate thresholds ``tau`` on that score and selects
        ``tau*`` that yields the highest expected net profit under the principled
        ``expected_profit`` model -- replacing the fixed rules (e.g. p > 0.7 or the
        80th percentile of p x CLV) with a data-learned cutoff. An optional
        ``budget`` caps the number of contacted customers (highest score first).

        Returns ``(sweep_df, best_row)`` where ``sweep_df`` contains one row per
        candidate threshold and ``best_row`` is the profit-maximising entry.
        """
        churn_probs = np.asarray(churn_probs, dtype=float)
        clv_values = np.asarray(clv_values, dtype=float)
        # Expected-profit contribution per customer (value-weighted score).
        score = success_rate * churn_probs * clv_values
        if thresholds is None:
            lo, hi = float(score.min()), float(score.max())
            if hi <= lo:
                thresholds = [lo]
            else:
                # Fine percentile grid over the score distribution so the
                # profit-maximising cutoff (near the per-customer cost breakeven)
                # is captured; a coarse linear grid can skip it.
                thresholds = np.percentile(score, np.linspace(0.5, 99.5, 100)).tolist()
        thresholds = [float(t) for t in thresholds]

        rows = []
        best = None
        for tau in thresholds:
            mask = score >= tau
            if budget is not None and intervention_cost > 0:
                allow = int(budget // intervention_cost)
                if int(mask.sum()) > allow:
                    sub_idx = np.where(mask)[0]
                    order = sub_idx[np.argsort(score[sub_idx])[::-1][:allow]]
                    mask = np.zeros_like(mask, dtype=bool)
                    mask[order] = True
            saved, cost, net, roi = expected_profit(
                churn_probs, clv_values, mask, intervention_cost, success_rate)
            row = {
                "threshold": round(tau, 4),
                "n_targeted": int(mask.sum()),
                "expected_revenue_saved": round(saved, 2),
                "total_cost": round(cost, 2),
                "net_benefit": round(net, 2),
                "roi_pct": round(roi, 2),
            }
            rows.append(row)
            # Only consider actionable (non-empty) thresholds as candidate optima;
            # the trivial "target nobody" solution (net=0) must not be selected.
            if int(mask.sum()) == 0:
                continue
            if best is None or net > best["net_benefit"]:
                best = dict(row)
        if best is None and rows:
            # Fallback: pick the least-lossy entry if every threshold is empty.
            best = max(rows, key=lambda r: r["net_benefit"])
        sweep = pd.DataFrame(rows)
        return sweep, best

    def _get_recommendation(self, roi: float, risk_reduction: float) -> str:
        if roi > 200:
            return "Highly recommended - strong ROI"
        elif roi > 50:
            return "Recommended - positive ROI"
        elif roi > 0:
            return "Marginally beneficial"
        else:
            return "Not recommended - negative ROI"
