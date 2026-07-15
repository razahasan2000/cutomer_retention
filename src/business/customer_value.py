import numpy as np
import pandas as pd
from typing import Optional


class CustomerLifetimeValue:
    def __init__(self, discount_rate: float = 0.1):
        self.discount_rate = discount_rate

    def compute_clv(self, avg_monthly_spend: np.ndarray,
                    tenure: np.ndarray, churn_prob: np.ndarray) -> np.ndarray:
        expected_months = (1 - churn_prob) * 12
        monthly_profit = avg_monthly_spend * 0.7
        clv = monthly_profit * (1 - (1 + self.discount_rate) ** -expected_months) / self.discount_rate
        clv = np.clip(clv, 0, None)
        return clv

    def compute_retention_value(self, clv: np.ndarray,
                                retention_cost: np.ndarray,
                                success_probability: np.ndarray) -> np.ndarray:
        expected_benefit = clv * success_probability
        net_value = expected_benefit - retention_cost
        roi = np.where(retention_cost > 0, (net_value / retention_cost) * 100, 0)
        return net_value, roi
