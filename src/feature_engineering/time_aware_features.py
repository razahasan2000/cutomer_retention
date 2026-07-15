import pandas as pd
import numpy as np
from typing import List


class TimeAwareFeatureEngineer:
    def __init__(self):
        self.created_features: List[str] = []

    def fit_transform(self, df: pd.DataFrame, tenure_col: str = "tenure",
                      monthly_col: str = "MonthlyCharges",
                      total_col: str = "TotalCharges",
                      contract_col: str = "Contract") -> pd.DataFrame:
        df = df.copy()

        df["AvgMonthlySpend"] = df[total_col] / (df[tenure_col] + 1)

        raw_accel = df[monthly_col] - df["AvgMonthlySpend"]
        lo = raw_accel.quantile(0.01)
        hi = raw_accel.quantile(0.99)
        df["ChargeAcceleration"] = raw_accel.clip(lower=lo, upper=hi)

        contract_risk_map = {"Month-to-month": 3, "One year": 2, "Two year": 1}
        if contract_col in df.columns:
            df["ContractRisk"] = df[contract_col].map(contract_risk_map).fillna(2)
        else:
            df["ContractRisk"] = 2

        service_cols = [
            "PhoneService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        present_services = [c for c in service_cols if c in df.columns]
        if present_services:
            df["ServiceCount"] = (df[present_services] == "Yes").sum(axis=1)
        else:
            df["ServiceCount"] = 0

        df["TenureDecile"] = pd.qcut(df[tenure_col] + 1, q=10, labels=False, duplicates="drop")

        self.created_features = [
            "AvgMonthlySpend", "ChargeAcceleration",
            "ContractRisk", "ServiceCount", "TenureDecile"
        ]
        return df

    def get_feature_names(self) -> List[str]:
        return self.created_features.copy()
