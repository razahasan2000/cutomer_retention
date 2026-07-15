import pandas as pd
from src.preprocessing.base_processor import BaseProcessor


class IranianProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.numeric_cols = [
            "Call Failure", "Subscription Length", "Seconds of Use",
            "Frequency of use", "Frequency of SMS", "Distinct Called Numbers",
            "Customer Value", "Charge Amount"
        ]
        self.categorical_cols = [
            "Complains", "Age Group", "Tariff Plan", "Status"
        ]
        self.target_col = "Churn"

    def load_data(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        import re
        df.columns = [re.sub(r'\s+', ' ', c).strip() for c in df.columns]
        if "Customer ID" in df.columns:
            df = df.drop(columns=["Customer ID"])
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df
