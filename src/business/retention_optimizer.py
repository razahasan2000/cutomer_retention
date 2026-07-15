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

    @staticmethod
    def _budget_mask(score: np.ndarray, mask: np.ndarray, intervention_cost: float,
                    budget: Optional[float]) -> np.ndarray:
        mask = mask.copy()
        if budget is not None and intervention_cost > 0:
            allow = int(budget // intervention_cost)
            if int(mask.sum()) > allow:
                sub = np.where(mask)[0]
                order = sub[np.argsort(score[sub])[::-1][:allow]]
                mask = np.zeros_like(mask, dtype=bool)
                mask[order] = True
        return mask

    def compare_thresholds(self, churn_probs: np.ndarray, clv_values: np.ndarray,
                           intervention_cost: float, success_rate: float = 0.3,
                           fixed_thresholds: List[float] = None, budget: Optional[float] = None,
                           n_bootstrap: int = 2000, seed: int = 42) -> Tuple[pd.DataFrame, Dict]:
        """Compare the learned adaptive threshold against fixed probability thresholds.

        For each fixed cutoff ``tau`` on the raw churn probability (target customers
        with ``p >= tau``) and for the adaptive optimum, expected net profit is
        computed with the principled ``expected_profit`` model. To demonstrate the
        adaptive profit gain is not a sampling artefact, two *paired* tests are
        reported per fixed threshold:
          * a paired bootstrap (B resamples of customers with replacement) of the
            per-customer profit difference (adaptive - fixed), yielding a 95% CI; and
          * a Wilcoxon signed-rank test on the non-zero per-customer differences.
        Returns ``(comparison_df, adaptive_summary)``.
        """
        churn_probs = np.asarray(churn_probs, dtype=float)
        clv_values = np.asarray(clv_values, dtype=float)
        score = success_rate * churn_probs * clv_values
        contrib = score - intervention_cost  # expected profit per targeted customer

        # Adaptive optimum (on the value-weighted expected-profit score)
        sweep, best = self.optimal_threshold(
            churn_probs, clv_values, intervention_cost, success_rate, budget)
        tau_star = best["threshold"]
        adapt_mask = self._budget_mask(score, score >= tau_star, intervention_cost, budget)
        adapt_profit = float(contrib[adapt_mask].sum())
        ind_adapt = adapt_mask.astype(int)

        if fixed_thresholds is None:
            fixed_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        fixed_thresholds = [float(t) for t in fixed_thresholds]

        n = len(churn_probs)
        rng = np.random.RandomState(seed)
        idx = rng.randint(0, n, size=(n_bootstrap, n))

        rows = []
        for tau in fixed_thresholds:
            fmask = self._budget_mask(score, churn_probs >= tau, intervention_cost, budget)
            ind_f = fmask.astype(int)
            fixed_profit = float(contrib[fmask].sum())
            n_f = int(fmask.sum())
            # Per-customer profit difference (adaptive - fixed)
            d = contrib * (ind_adapt - ind_f)
            boot = d[idx].sum(axis=1)
            ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
            nz = d[d != 0]
            w_p = None
            if len(nz) >= 2 and np.any(nz != 0):
                try:
                    from scipy.stats import wilcoxon
                    _, w_p = wilcoxon(nz)
                except Exception:
                    w_p = None
            delta = adapt_profit - fixed_profit
            rows.append({
                "threshold": round(tau, 3),
                "type": "fixed_p",
                "n_targeted": n_f,
                "net_profit": round(fixed_profit, 2),
                "roi_pct": round((fixed_profit / (n_f * intervention_cost) * 100) if n_f > 0 else 0, 2),
                "delta_vs_adaptive": round(delta, 2),
                "ci_2_5": round(ci_low, 2),
                "ci_97_5": round(ci_high, 2),
                "wilcoxon_p": round(float(w_p), 4) if w_p is not None else None,
                "significant": bool(ci_low > 0) or (w_p is not None and w_p < 0.05),
            })
        summary = {
            "adaptive_threshold": round(tau_star, 4),
            "adaptive_n_targeted": int(adapt_mask.sum()),
            "adaptive_net_profit": round(adapt_profit, 2),
            "adaptive_roi_pct": round((adapt_profit / (int(adapt_mask.sum()) * intervention_cost) * 100)
                                     if adapt_mask.sum() > 0 else 0, 2),
            "n_bootstrap": n_bootstrap,
        }
        return pd.DataFrame(rows), summary

    def _get_recommendation(self, roi: float, risk_reduction: float) -> str:
        if roi > 200:
            return "Highly recommended - strong ROI"
        elif roi > 50:
            return "Recommended - positive ROI"
        elif roi > 0:
            return "Marginally beneficial"
        else:
            return "Not recommended - negative ROI"
