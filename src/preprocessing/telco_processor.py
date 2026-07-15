import pandas as pd
import numpy as np
from src.preprocessing.base_processor import BaseProcessor


class TelcoProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        self.categorical_cols = [
            "gender", "SeniorCitizen", "Partner", "Dependents",
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
            "Contract", "PaperlessBilling", "PaymentMethod"
        ]
        self.target_col = "Churn"

    def load_data(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
        df[self.target_col] = df[self.target_col].map({"Yes": 1, "No": 0})
        return df
